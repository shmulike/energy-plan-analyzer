# streamlit_power_dashboard_dark_apple_colors_locked.py
import io, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============ Config ============
DEFAULT_ELECTRIC_PRICE = 0.6402  # NIS/kWh
PLOT_WIDTH_INCHES  = 12
PLOT_HEIGHT_INCHES = 6
AXIS_TEXT_SIZE     = 10
LEGEND_TEXT_SIZE   = 10
NIGHT_START, NIGHT_END = 23, 7   # plot night window
MAX_PLANS = 5

# Apple-style default plot colors (dark mode friendly)
DEFAULT_DAY_COLOR   = "#0A84FF"  # Apple Blue
DEFAULT_NIGHT_COLOR = "#5E5CE6"  # Apple Indigo

# Master switch: allow users to change plot colors?
ALLOW_USER_COLOR_PICKER = False  # <- default: False (no color controls shown)

st.set_page_config(page_title="Electricity Consumption Dashboard", layout="wide")

# ---------- Intro ----------
st.markdown(
    """
<div style="margin:0 0 0.3rem 0;">
  <h3 style="margin:0;color:#e6edf3;">Electricity Consumption Dashboard</h3>
  <p style="margin:.2rem 0 0 0;color:#8b949e;">
    Created by <strong>Shmulik Edelman</strong>. Upload your CSV, visualize day vs. night consumption, and compare up to five pricing plans.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Dark theme ----------
st.markdown(
    """
<style>
:root{
  --bg:#0f1115;--card:#161b22;--muted:#8b949e;--border:#30363d;--text:#e6edf3;
  --accent:#1f6feb;--accent-hover:#1a60d1;--row-green:#163e29;--row-green-text:#d1f7d6;
}
.stApp{background-color:var(--bg);color:var(--text);}
.block-container{padding-top:.6rem;padding-bottom:.6rem;}
.card{background:var(--card);border-radius:16px;padding:1rem;border:1px solid var(--border);
      box-shadow:0 8px 24px rgba(0,0,0,.25);}
h1,h2,h3,h4,h5{color:var(--text);margin:0 0 .6rem 0;}
input,textarea,select,.stTextInput input,.stNumberInput input,
.stSelectbox div[role="button"],.stTextArea textarea{
  background:#0d1117!important;color:var(--text)!important;border:1px solid var(--border)!important;
  border-radius:12px!important;
}
.stNumberInput button,.stSelectbox svg{color:var(--text)!important;}
.stButton>button{background:var(--accent);color:#fff;border-radius:12px;border:none;
  padding:.45rem .9rem;font-weight:600;box-shadow:0 6px 16px rgba(31,111,235,.35);}
.stButton>button:hover{background:var(--accent-hover);}
</style>
""",
    unsafe_allow_html=True,
)

# ============ Helpers ============
def find_header_line(raw_bytes: bytes) -> int:
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = raw_bytes.decode("cp1255", errors="replace")
    for i, line in enumerate(text.splitlines()[:300]):
        if ("תאריך" in line and "מועד" in line) or ("Date" in line and "Time" in line):
            return i
    return 0

def load_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["datetime","consumption_kwh"])
    raw = file.read()
    header_line = find_header_line(raw)
    df = None
    for enc in ["utf-8","cp1255"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), skiprows=header_line, encoding=enc, quotechar='"')
            if not df.empty: break
        except Exception: df = None
    if df is None or df.empty: return pd.DataFrame(columns=["datetime","consumption_kwh"])
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {"date":None,"time":None,"cons":None}
    candidates = {
        "date":["תאריך","Date"], "time":["מועד תחילת הפעימה","Time","שעה"],
        "cons":['צריכה בקוט"ש','צריכה בקוט""ש',"צריכה בקוטש","Consumption (kWh)","kWh","צריכה"]
    }
    for k, names in candidates.items():
        for n in names:
            for c in df.columns:
                if n == c or re.sub(r"\s+","",n) == re.sub(r"\s+","",c):
                    col_map[k] = c; break
            if col_map[k]: break

    if col_map["date"] is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(5).tolist())
            if "/" in sample or "-" in sample: col_map["date"]=c; break
    if col_map["time"] is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(5).tolist())
            if ":" in sample: col_map["time"]=c; break
    if col_map["cons"] is None:
        for c in df.columns:
            try: pd.to_numeric(df[c], errors="raise"); col_map["cons"]=c; break
            except Exception: continue

    if None in col_map.values(): return pd.DataFrame(columns=["datetime","consumption_kwh"])
    cons = pd.to_numeric(df[col_map["cons"]], errors="coerce")
    dt = pd.to_datetime(df[col_map["date"]].astype(str)+" "+df[col_map["time"]].astype(str),
                        dayfirst=True, errors="coerce")
    out = pd.DataFrame({"datetime":dt,"consumption_kwh":cons})
    return out.dropna(subset=["datetime","consumption_kwh"]).sort_values("datetime").reset_index(drop=True)

def split_day_night(df, start_hour=NIGHT_START, end_hour=NIGHT_END):
    h = df["datetime"].dt.hour
    return (h >= start_hour) | (h < end_hour)

def aggregate(df, granularity:str):
    if df.empty: return pd.DataFrame(columns=["label","day_kwh","night_kwh"])
    work = df.copy(); work["is_night"] = split_day_night(work).astype(int)
    if granularity == "Week":
        iso = work["datetime"].dt.isocalendar()
        work["period"] = iso.year.astype(str)+"-W"+iso.week.astype(str)
    else:
        work["period"] = work["datetime"].dt.to_period("M").astype(str)
    grp = work.groupby(["period","is_night"])["consumption_kwh"].sum().unstack(fill_value=0)
    grp.columns = ["day_kwh","night_kwh"]
    return grp.reset_index().rename(columns={"period":"label"})

def plot_stacked(df_agg, title, color_day, color_night):
    if df_agg.empty:
        st.warning("No data to plot."); return
    labels = df_agg["label"].tolist()
    day = df_agg["day_kwh"].values; night = df_agg["night_kwh"].values
    fig = plt.figure(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('#0f1115'); ax.set_facecolor('#0f1115')
    for s in ax.spines.values(): s.set_color('#8b949e')
    x = np.arange(len(labels))
    ax.bar(x, day,   label="Day",   color=color_day)
    ax.bar(x, night, bottom=day, label="Night", color=color_night)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=AXIS_TEXT_SIZE, color='#e6edf3')
    ax.set_xlabel("Period", fontsize=AXIS_TEXT_SIZE, color='#e6edf3')
    ax.set_ylabel("Consumption (kWh)", fontsize=AXIS_TEXT_SIZE, color='#e6edf3')
    ax.set_title(title, fontsize=AXIS_TEXT_SIZE+2, color='#e6edf3')
    ax.legend(fontsize=LEGEND_TEXT_SIZE, facecolor='#161b22', edgecolor='#30363d')
    ax.tick_params(colors='#e6edf3'); fig.tight_layout()
    st.pyplot(fig)

def compute_cost(df, price_kwh, mode, discount_pct, start_hour=0, end_hour=0):
    if df.empty: return 0.0
    try:
        price_kwh = float(price_kwh); discount_pct = float(discount_pct)
        start_hour = int(start_hour); end_hour = int(end_hour)
    except Exception: return 0.0
    cons = df.copy(); cons["hour"] = cons["datetime"].dt.hour
    total = cons["consumption_kwh"].sum()
    if mode == "All day":
        return total * price_kwh * (1 - discount_pct/100.0)
    if start_hour == end_hour:
        mask = np.ones(len(cons), dtype=bool)
    elif start_hour < end_hour:
        mask = (cons["hour"] >= start_hour) & (cons["hour"] < end_hour)
    else:
        mask = (cons["hour"] >= start_hour) | (cons["hour"] < end_hour)
    win = cons.loc[mask,"consumption_kwh"].sum(); rest = total - win
    return rest*price_kwh + win*price_kwh*(1 - discount_pct/100.0)

# ============ Session (visibility of extra plans) ============
if "plan4_visible" not in st.session_state: st.session_state.plan4_visible = False
if "plan5_visible" not in st.session_state: st.session_state.plan5_visible = False

# ============ TOP: Controls + Chart ============
left, right = st.columns([4,8], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload CSV"); uploaded = st.file_uploader("Electricity CSV", type=["csv"])
    st.subheader("Aggregation"); granularity = st.radio("Choose", ["Week","Month"], horizontal=True, index=1)

    # Colors section: only if allowed; otherwise enforce defaults silently
    if ALLOW_USER_COLOR_PICKER:
        st.subheader("Colors")
        default_day  = st.toggle("Use default Day color (Apple Blue)", value=True)
        day_color    = DEFAULT_DAY_COLOR if default_day else st.color_picker("Day bar color", DEFAULT_DAY_COLOR, key="c_day")
        default_night= st.toggle("Use default Night color (Apple Indigo)",  value=True)
        night_color  = DEFAULT_NIGHT_COLOR if default_night else st.color_picker("Night bar color", DEFAULT_NIGHT_COLOR, key="c_night")
    else:
        day_color, night_color = DEFAULT_DAY_COLOR, DEFAULT_NIGHT_COLOR

    st.subheader("Reference price")
    ref_price = st.number_input("Electric price per kWh (NIS)", value=DEFAULT_ELECTRIC_PRICE, step=0.01, format="%.4f")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dashboard")
    data = load_csv(uploaded)
    if not data.empty:
        st.caption(f"Loaded {len(data):,} rows. Range: {data['datetime'].min().date()} → {data['datetime'].max().date()}")
    else:
        st.info("Upload a CSV to see plot and pricing.")
    agg_df = aggregate(data, granularity)
    plot_stacked(agg_df, f"Consumption ({granularity}) - Day vs Night", day_color, night_color)
    st.markdown('</div>', unsafe_allow_html=True)

# ============ BOTTOM: Full‑width Plans + Cost Comparison ============
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Pricing Plans")

# Columns in order 1..5 from left to right:
cols = st.columns(5, gap="large")
plans = []

def plan_card(col, idx, default_mode="All day", default_price=None, default_discount=0.0,
              default_start=0, default_end=0):
    if default_price is None: default_price = ref_price
    with col:
        st.markdown(f"**Plan {idx}**")
        mode = st.selectbox(f"Mode P{idx}", ["All day","By hour"],
                            index=0 if default_mode=="All day" else 1, key=f"mode_{idx}")
        price_val = st.number_input(f"Price P{idx} (NIS/kWh)", value=float(default_price),
                                    step=0.01, format="%.4f", key=f"price_{idx}")
        discount = st.number_input(f"Discount % P{idx}", value=float(default_discount),
                                   step=0.1, format="%.1f", key=f"disc_{idx}")
        if mode == "By hour":
            c1, c2 = st.columns(2)
            with c1:
                start_h = st.number_input("Start", min_value=0, max_value=23,
                                          value=int(default_start), step=1, key=f"start_{idx}")
            with c2:
                end_h   = st.number_input("End",   min_value=0, max_value=23,
                                          value=int(default_end), step=1, key=f"end_{idx}")
        else:
            start_h, end_h = 0, 0
        return (mode, price_val, discount, start_h, end_h)

# Plans 1..3 (always visible) in cols[0..2]
plans.append(plan_card(cols[0], 1, default_mode="All day", default_discount=6.0))
plans.append(plan_card(cols[1], 2, default_mode="By hour", default_discount=20.0, default_start=23, default_end=7))
plans.append(plan_card(cols[2], 3, default_mode="All day", default_discount=0.0))

# cols[3] -> Plan 4 placeholder or Plan 4
with cols[3]:
    if not st.session_state.plan4_visible:
        if st.button("＋ Add Plan 4"):
            st.session_state.plan4_visible = True
    else:
        plans.append(plan_card(cols[3], 4, default_mode="All day"))

# cols[4] -> Plan 5 placeholder or Plan 5
with cols[4]:
    if not st.session_state.plan5_visible:
        if st.button("＋ Add Plan 5"):
            st.session_state.plan5_visible = True
    else:
        plans.append(plan_card(cols[4], 5, default_mode="All day"))

st.markdown("---")
st.subheader("Cost Comparison")

if not data.empty:
    total_kwh = data["consumption_kwh"].sum()
    base_cost = total_kwh * ref_price
    rows = [("Reference (no discount)", base_cost, 0.0)]
    for i, p in enumerate(plans, start=1):
        c = compute_cost(data, p[1], p[0], p[2], p[3], p[4])
        rows.append((f"Plan {i}", c, base_cost - c))
    df_costs = pd.DataFrame(rows, columns=["Plan","Total cost (NIS)","Savings vs ref. (NIS)"])
    # one decimal everywhere
    df_costs["Total cost (NIS)"]      = df_costs["Total cost (NIS)"].astype(float)
    df_costs["Savings vs ref. (NIS)"] = df_costs["Savings vs ref. (NIS)"].astype(float)

    # winner (only among "Plan *" rows)
    plan_rows = df_costs[df_costs["Plan"].str.startswith("Plan")]
    min_cost = plan_rows["Total cost (NIS)"].min() if not plan_rows.empty else np.inf

    base_styles = [
        {"selector":"th","props":[("background-color","#0d1117"),("color","#e6edf3"),("border","1px solid #30363d")]},
        {"selector":"td","props":[("background-color","#0d1117"),("color","#e6edf3"),("border","1px solid #30363d")]},
        {"selector":"tr","props":[("background-color","#0d1117")]}
    ]

    def highlight_best(row):
        if row["Plan"].startswith("Plan") and float(row["Total cost (NIS)"]) == float(min_cost):
            return ['background-color: var(--row-green); color: var(--row-green-text); font-weight: 600']*len(row)
        return ['']*len(row)

    styled = (
        df_costs
        .style
        .set_table_styles(base_styles)
        .format({"Total cost (NIS)":"{:.1f}", "Savings vs ref. (NIS)":"{:.1f}"})
        .apply(highlight_best, axis=1)
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.info("No data to compute costs.")

st.markdown('</div>', unsafe_allow_html=True)
