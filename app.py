# app.py — Energy Plan Analyzer (Light/Dark switch, fixed table highlight, fixed plot legend)
import io, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ===================== Config =====================
DEFAULT_ELECTRIC_PRICE = 0.6402  # NIS/kWh
PLOT_WIDTH_INCHES  = 12
PLOT_HEIGHT_INCHES = 6
AXIS_TEXT_SIZE     = 10
LEGEND_TEXT_SIZE   = 10
NIGHT_START, NIGHT_END = 23, 7
MAX_PLANS = 5

# Default plot colors (used in BOTH themes unless you change them)
DARK_MODE_DAY_COLOR    = "#6200EE"
DARK_MODE_NIGHT_COLOR  = "#03DAC6"
LIGHT_MODE_DAY_COLOR   = "#6200EE"
LIGHT_MODE_NIGHT_COLOR = "#03DAC6"

ALLOW_USER_COLOR_PICKER = False  # keep color pickers hidden

# ===================== Page setup =====================
st.set_page_config(page_title="Electricity Consumption Dashboard", layout="wide")

# ===================== Theme state + visible switch =====================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # default to dark

top_l, top_sp, top_r = st.columns([9, 1, 2])
with top_l:
    st.markdown(
        """
<div style="margin:0 0 .3rem 0;">
  <h3 style="margin:0;">Electricity Consumption Dashboard</h3>
  <p style="margin:.2rem 0 0 0;opacity:.8;">
    Created by <strong>Shmulik Edelman</strong>. Upload your CSV, visualize day vs. night usage, and compare up to five pricing plans.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
with top_r:
    # Visible theme switch (toggle if available, else checkbox fallback)
    if hasattr(st, "toggle"):
        dark_choice = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode, key="__dark_toggle__")
    else:
        dark_choice = st.checkbox("🌙 Dark mode", value=st.session_state.dark_mode, key="__dark_checkbox__")
    st.session_state.dark_mode = bool(dark_choice)

# ===================== Theming (CSS) =====================
if st.session_state.dark_mode:
    # Palette for Dark
    BG        = "#0f1115"
    CARD      = "#161b22"
    MUTED     = "#8b949e"
    BORDER    = "#30363d"
    TEXT      = "#e6edf3"
    ACCENT    = "#0A84FF"
    ACCENT_H  = "#066de6"
    DF_TH_BG  = CARD
    DF_TD_BG  = CARD
    DF_TEXT   = TEXT
    WIN_BG    = "#1f5136"  # <- winner row background (dark green)
    WIN_FG    = "#ffffff"  # <- winner row text
else:
    # Palette for Light
    BG        = "#f5f5f7"
    CARD      = "#ffffff"
    MUTED     = "#6e6e73"
    BORDER    = "#e5e5ea"
    TEXT      = "#1d1d1f"
    ACCENT    = "#0A84FF"
    ACCENT_H  = "#0066d6"
    DF_TH_BG  = CARD
    DF_TD_BG  = CARD
    DF_TEXT   = TEXT
    WIN_BG    = "#c8f3dc"  # <- winner row background (mint)
    WIN_FG    = "#083c2a"  # <- winner row text (deep green)

st.markdown(
    f"""
<style>
.stApp{{background-color:{BG};color:{TEXT};}}
.block-container{{padding-top:.6rem;padding-bottom:.6rem;}}
.card{{background:{CARD};border-radius:16px;padding:1rem;border:1px solid {BORDER};
      box-shadow:0 8px 24px rgba(0,0,0,.12);}}
h1,h2,h3,h4,h5{{color:{TEXT};margin:0 0 .6rem 0;}}

/* inputs */
input,textarea,select,.stTextInput input,.stNumberInput input,
.stSelectbox div[role="button"],.stTextArea textarea{{
  background:transparent!important;color:{TEXT}!important;border:1px solid {BORDER}!important;
  border-radius:12px!important;
}}
.stNumberInput button,.stSelectbox svg{{color:{TEXT}!important;}}

/* buttons */
.stButton>button{{background:{ACCENT};color:#fff;border-radius:12px;border:none;
  padding:.45rem .9rem;font-weight:600;box-shadow:0 6px 16px rgba(10,132,255,.25);}}
.stButton>button:hover{{background:{ACCENT_H};}}

/* dataframe base to avoid odd backgrounds */
.dataframe th{{background:{DF_TH_BG}!important;color:{DF_TEXT}!important;border:1px solid {BORDER}!important;}}
.dataframe td{{background:{DF_TD_BG}!important;color:{DF_TEXT}!important;border:1px solid {BORDER}!important;}}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== Helpers =====================
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

def plot_stacked(df_agg, title, day_color, night_color, dark: bool):
    if df_agg.empty:
        st.warning("No data to plot."); return
    labels = df_agg["label"].tolist()
    day = df_agg["day_kwh"].values; night = df_agg["night_kwh"].values

    fig = plt.figure(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    ax = fig.add_subplot(111)
    if dark:
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        axis_color = TEXT; spine_color = MUTED
        legend_face = CARD; legend_edge = BORDER; legend_text = "#FFFFFF"
    else:
        fig.patch.set_facecolor(BG); ax.set_facecolor("#FFFFFF")
        axis_color = TEXT; spine_color = "#d2d2d7"
        legend_face = "#FFFFFF"; legend_edge = BORDER; legend_text = "#1d1d1f"

    for s in ax.spines.values(): s.set_color(spine_color)
    x = np.arange(len(labels))
    ax.bar(x, day,   label="Day",   color=day_color)
    ax.bar(x, night, bottom=day, label="Night", color=night_color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=AXIS_TEXT_SIZE, color=axis_color)
    ax.set_xlabel("Period", fontsize=AXIS_TEXT_SIZE, color=axis_color)
    ax.set_ylabel("Consumption (kWh)", fontsize=AXIS_TEXT_SIZE, color=axis_color)
    ax.set_title(title, fontsize=AXIS_TEXT_SIZE+2, color=axis_color)

    leg = ax.legend(fontsize=LEGEND_TEXT_SIZE, facecolor=legend_face, edgecolor=legend_edge)
    for txt in leg.get_texts(): txt.set_color(legend_text)  # ensure contrast

    ax.tick_params(colors=axis_color)
    fig.tight_layout()
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

# ===================== Session for extra plans =====================
if "plan4_visible" not in st.session_state: st.session_state.plan4_visible = False
if "plan5_visible" not in st.session_state: st.session_state.plan5_visible = False

# ===================== TOP: Controls + Chart =====================
left, right = st.columns([4,8], gap="large")

with left:
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Electricity CSV", type=["csv"])

    st.subheader("Aggregation")
    granularity = st.radio("Choose", ["Week","Month"], horizontal=True, index=1)

    # Theme-derived plot colors (no user picker)
    if st.session_state.dark_mode:
        day_color, night_color = DARK_MODE_DAY_COLOR, DARK_MODE_NIGHT_COLOR
    else:
        day_color, night_color = LIGHT_MODE_DAY_COLOR, LIGHT_MODE_NIGHT_COLOR

    if ALLOW_USER_COLOR_PICKER:
        st.info("Color picker disabled by configuration.")

    st.subheader("Reference price")
    ref_price = st.number_input("Electric price per kWh (NIS)", value=DEFAULT_ELECTRIC_PRICE, step=0.01, format="%.4f")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    st.subheader("Dashboard")
    data = load_csv(uploaded)
    if not data.empty:
        st.caption(f"Loaded {len(data):,} rows. Range: {data['datetime'].min().date()} → {data['datetime'].max().date()}")
    else:
        st.info("Upload a CSV to see plot and pricing.")
    agg_df = aggregate(data, granularity)
    plot_stacked(agg_df, f"Consumption ({granularity}) - Day vs Night",
                 day_color, night_color, dark=st.session_state.dark_mode)
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== BOTTOM: Full‑width Plans + Cost Comparison =====================
st.markdown(f'<div class="card">', unsafe_allow_html=True)
st.subheader("Pricing Plans")

cols = st.columns(5, gap="large")
plans = []

def plan_card(col, idx, default_mode="All day", default_price=None, default_discount=0.0,
              default_start=0, default_end=0, allow_remove=False):
    if default_price is None: default_price = ref_price
    with col:
        if allow_remove:
            top = st.columns([1,1])
            with top[0]:
                st.markdown(f"**Plan {idx}**")
            with top[1]:
                if st.button(f"Remove {idx}", key=f"rm_{idx}"):
                    st.session_state[f"plan{idx}_visible"] = False
                    st.experimental_rerun()
        else:
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

# Plans 1..3 (always visible) left -> right
plans.append(plan_card(cols[0], 1, default_mode="All day", default_discount=6.0))
plans.append(plan_card(cols[1], 2, default_mode="By hour", default_discount=20.0, default_start=23, default_end=7))
plans.append(plan_card(cols[2], 3, default_mode="All day", default_discount=0.0))

# Plan 4
with cols[3]:
    if not st.session_state.plan4_visible:
        if st.button("＋ Add Plan 4", key="add4"):
            st.session_state.plan4_visible = True
            st.experimental_rerun()
    else:
        plans.append(plan_card(cols[3], 4, default_mode="All day", allow_remove=True))

# Plan 5
with cols[4]:
    if not st.session_state.plan5_visible:
        if st.button("＋ Add Plan 5", key="add5"):
            st.session_state.plan5_visible = True
            st.experimental_rerun()
    else:
        plans.append(plan_card(cols[4], 5, default_mode="All day", allow_remove=True))

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

    # cheapest among actual plans (ignore Reference)
    plan_rows = df_costs[df_costs["Plan"].str.startswith("Plan")]
    min_cost = plan_rows["Total cost (NIS)"].min() if not plan_rows.empty else np.inf

    # Base table styles with fixed HEX colors (no CSS variables inside the Styler)
    base_styles = [
        {"selector":"th","props":[("background-color", DF_TH_BG),("color", DF_TEXT),("border", f"1px solid {BORDER}")]},
        {"selector":"td","props":[("background-color", DF_TD_BG),("color", DF_TEXT),("border", f"1px solid {BORDER}")]},
        {"selector":"tr","props":[("background-color", DF_TD_BG)]},
    ]

    # WIN_CELL  = f"background-color: {WIN_BG} !important; color: {WIN_FG} !important; font-weight: 700 !important;"
    WIN_CELL = "background-color: #90EE90 !important; color: #000000 !important; font-weight: 700 !important;"

    def highlight_best(row):
        if row["Plan"].startswith("Plan") and float(row["Total cost (NIS)"]) == float(min_cost):
            return [WIN_CELL] * len(row)
        return [''] * len(row)

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
