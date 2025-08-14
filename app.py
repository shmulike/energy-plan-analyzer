# streamlit_power_dashboard_dynamic_plans_dark_fullwidth.py

import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =============================
# Configurable Defaults
# =============================
DEFAULT_ELECTRIC_PRICE = 0.6402  # NIS per kWh
PLOT_WIDTH_INCHES = 12
PLOT_HEIGHT_INCHES = 6
AXIS_TEXT_SIZE = 10
LEGEND_TEXT_SIZE = 10
NIGHT_START = 23
NIGHT_END = 7  # exclusive
MAX_PLANS = 5  # absolute max; starts with 3 visible

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Electricity Consumption Dashboard", layout="wide")

# =============================
# Intro text
# =============================
st.markdown("""
### Electricity Consumption Dashboard
Created by **Shmulik Edelman**, this tool helps you upload and analyze your household or business electricity usage.  
Compare up to five custom pricing plans, visualize day vs. night consumption, and instantly see which plan saves you the most money.
""")

# =============================
# Dark theme styling (widgets + panels)
# =============================
st.markdown("""
    <style>
        :root {
            --bg: #0f1115;
            --card: #161b22;
            --muted: #8b949e;
            --border: #30363d;
            --text: #e6edf3;
            --accent: #1f6feb;
            --accent-hover: #1a60d1;
            --row-green: #163e29;
            --row-green-text: #d1f7d6;
        }
        .stApp { background-color: var(--bg); color: var(--text); }
        h1, h2, h3, h4, h5 { color: var(--text); }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }

        /* Card-like columns */
        .card {
            background: var(--card);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            border: 1px solid var(--border);
        }

        /* Inputs (dark) */
        input, textarea, select, .stTextInput input, .stNumberInput input,
        .stSelectbox div[role="button"], .stTextArea textarea {
            background-color: #0d1117 !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }
        .stNumberInput button, .stSelectbox svg { color: var(--text) !important; }

        /* Buttons */
        .stButton>button {
            background: var(--accent); color: #fff; border-radius: 12px;
            border: none; padding: 0.5rem 0.9rem; font-weight: 600;
            box-shadow: 0 6px 16px rgba(31,111,235,0.35);
        }
        .stButton>button:hover { background: var(--accent-hover); }

        /* Dataframe rows */
        .dataframe tbody tr { background: #0d1117; color: var(--text); }
        .dataframe tbody tr:hover { background: #11151c; }
    </style>
""", unsafe_allow_html=True)

# =============================
# Helpers
# =============================
def find_header_line(raw_bytes: bytes) -> int:
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = raw_bytes.decode("cp1255", errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines[:300]):
        if ("תאריך" in line and "מועד" in line) or ("Date" in line and "Time" in line):
            return i
    return 0

def load_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["datetime", "consumption_kwh"])
    raw = file.read()
    header_line = find_header_line(raw)
    df = None
    for enc in ["utf-8", "cp1255"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), skiprows=header_line, encoding=enc, quotechar='"')
            if not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "consumption_kwh"])
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {"date": None, "time": None, "cons": None}
    candidates = {
        "date": ["תאריך", "Date"],
        "time": ["מועד תחילת הפעימה", "Time", "שעה"],
        "cons": ['צריכה בקוט"ש', 'צריכה בקוט""ש', "צריכה בקוטש", "Consumption (kWh)", "kWh", "צריכה"]
    }
    for key, names in candidates.items():
        for n in names:
            for c in df.columns:
                if n == c or re.sub(r'\\s+', '', n) == re.sub(r'\\s+', '', c):
                    col_map[key] = c; break
            if col_map[key]: break

    if col_map["date"] is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(5).tolist())
            if "/" in sample or "-" in sample: col_map["date"] = c; break
    if col_map["time"] is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(5).tolist())
            if ":" in sample: col_map["time"] = c; break
    if col_map["cons"] is None:
        for c in df.columns:
            try:
                pd.to_numeric(df[c], errors="raise"); col_map["cons"] = c; break
            except Exception:
                continue

    if None in col_map.values():
        return pd.DataFrame(columns=["datetime", "consumption_kwh"])

    cons = pd.to_numeric(df[col_map["cons"]], errors="coerce")
    dt = pd.to_datetime(df[col_map["date"]].astype(str) + " " + df[col_map["time"]].astype(str),
                        dayfirst=True, errors="coerce")
    out = pd.DataFrame({"datetime": dt, "consumption_kwh": cons})
    return out.dropna(subset=["datetime", "consumption_kwh"]).sort_values("datetime").reset_index(drop=True)

def split_day_night(df: pd.DataFrame, start_hour=NIGHT_START, end_hour=NIGHT_END) -> pd.Series:
    hrs = df["datetime"].dt.hour
    return (hrs >= start_hour) | (hrs < end_hour)

def aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["label","day_kwh","night_kwh"])
    is_night = split_day_night(df)
    work = df.copy()
    work["is_night"] = is_night.astype(int)
    if granularity == "Week":
        work["year"] = work["datetime"].dt.isocalendar().year
        work["week"] = work["datetime"].dt.isocalendar().week
        work["period"] = work["year"].astype(str) + "-W" + work["week"].astype(str)
    else:
        work["period"] = work["datetime"].dt.to_period("M").astype(str)
    grp = work.groupby(["period","is_night"])["consumption_kwh"].sum().unstack(fill_value=0)
    grp.columns = ["day_kwh","night_kwh"]
    return grp.reset_index().rename(columns={"period":"label"})

def plot_stacked(df_agg: pd.DataFrame, title: str, color_day: str, color_night: str):
    if df_agg.empty:
        st.warning("No data to plot.")
        return
    labels = df_agg["label"].tolist()
    day = df_agg["day_kwh"].values
    night = df_agg["night_kwh"].values

    fig = plt.figure(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('#0f1115')
    ax.set_facecolor('#0f1115')
    for spine in ax.spines.values():
        spine.set_color('#8b949e')

    x = np.arange(len(labels))
    # Use chosen colors:
    ax.bar(x, day, label="Day", color=color_day or "#87CEEB")
    ax.bar(x, night, bottom=day, label="Night", color=color_night or "#000080")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=AXIS_TEXT_SIZE, color='#e6edf3')
    ax.set_xlabel("Period", fontsize=AXIS_TEXT_SIZE, color='#e6edf3')
    ax.set_ylabel("Consumption (kWh)", fontsize=AXIS_TEXT_SIZE, color='#e6edf3')
    ax.set_title(title, fontsize=AXIS_TEXT_SIZE+2, color='#e6edf3')
    ax.legend(fontsize=LEGEND_TEXT_SIZE, facecolor='#161b22', edgecolor='#30363d')
    ax.tick_params(colors='#e6edf3')
    fig.tight_layout()
    st.pyplot(fig)

def compute_cost(df: pd.DataFrame, price_kwh: float, mode: str, discount_pct: float,
                 start_hour: int = 0, end_hour: int = 0) -> float:
    if df.empty:
        return 0.0
    try:
        price_kwh = float(price_kwh)
        discount_pct = float(discount_pct)
        start_hour = int(start_hour)
        end_hour = int(end_hour)
    except Exception:
        return 0.0

    cons = df.copy()
    cons["hour"] = cons["datetime"].dt.hour
    total = cons["consumption_kwh"].sum()

    if mode == "All day":
        return total * price_kwh * (1 - discount_pct/100)
    else:
        if start_hour == end_hour:
            mask = np.ones(len(cons), dtype=bool)
        elif start_hour < end_hour:
            mask = (cons["hour"] >= start_hour) & (cons["hour"] < end_hour)
        else:
            mask = (cons["hour"] >= start_hour) | (cons["hour"] < end_hour)
        win = cons.loc[mask, "consumption_kwh"].sum()
        rest = total - win
        return rest * price_kwh + win * price_kwh * (1 - discount_pct/100)

# =============================
# Session state for plan visibility
# =============================
if "plan4_visible" not in st.session_state:
    st.session_state.plan4_visible = False
if "plan5_visible" not in st.session_state:
    st.session_state.plan5_visible = False

# =============================
# TOP: Controls + Chart (two columns)
# =============================
left, right = st.columns([4, 8], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Electricity CSV", type=["csv"])

    st.subheader("Aggregation")
    granularity = st.radio("Choose", ["Week", "Month"], horizontal=True, index=1)

    st.subheader("Colors")
    default_day = st.toggle("Default Day color (skyblue)", value=True)
    day_color = "#87CEEB" if default_day else st.color_picker("Day bar color", "#87CEEB")
    default_night = st.toggle("Default Night color (navy)", value=True)
    night_color = "#000080" if default_night else st.color_picker("Night bar color", "#000080")

    st.subheader("Reference price")
    ref_price = st.number_input("Electric price per kWh (NIS)", value=DEFAULT_ELECTRIC_PRICE, step=0.01, format="%.4f")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Dashboard")
    data = load_csv(uploaded)
    if not data.empty:
        st.caption(f"Loaded {len(data):,} rows. Range: {data['datetime'].min().date()} → {data['datetime'].max().date()}")
    else:
        st.info("Upload a CSV to see plot and pricing.")
    agg_df = aggregate(data, granularity)
    plot_stacked(agg_df, f"Consumption ({granularity}) - Day vs Night", day_color, night_color)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# BOTTOM: Full-width Plans + Cost Comparison
# =============================
st.markdown("## ")
bottom = st.container()
with bottom:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Pricing Plans (Full Width)")

    # 5 columns total: [Add4] [Add5] [Plan1] [Plan2] [Plan3]
    cols = st.columns(5, gap="large")

    # ----- Placeholders on the LEFT to reveal Plan 4 / Plan 5 -----
    with cols[0]:
        if not st.session_state.plan4_visible:
            st.markdown("### ")
            st.markdown("#### ")
            st.markdown("**Add Plan 4**")
            if st.button("＋", key="add4"):
                st.session_state.plan4_visible = True

    with cols[1]:
        if not st.session_state.plan5_visible:
            st.markdown("### ")
            st.markdown("#### ")
            st.markdown("**Add Plan 5**")
            if st.button("＋", key="add5"):
                st.session_state.plan5_visible = True

    plans = []

    # Helper to render one plan column
    def plan_card(col, idx, default_mode="All day", default_price=None, default_discount=0.0,
                  default_start=0, default_end=0):
        if default_price is None:
            default_price = ref_price
        with col:
            st.markdown(f"**Plan {idx}**")
            mode = st.selectbox(f"Mode P{idx}", ["All day", "By hour"], index=0 if default_mode=="All day" else 1, key=f"mode_{idx}")
            price_val = st.number_input(f"Price P{idx} (NIS/kWh)", value=float(default_price), step=0.01, format="%.4f", key=f"price_{idx}")
            discount = st.number_input(f"Discount % P{idx}", value=float(default_discount), step=0.1, format="%.1f", key=f"disc_{idx}")
            if mode == "By hour":
                sh_col, eh_col = st.columns(2)
                with sh_col:
                    start_h = st.number_input("Start", min_value=0, max_value=23, value=int(default_start), step=1, key=f"start_{idx}")
                with eh_col:
                    end_h = st.number_input("End", min_value=0, max_value=23, value=int(default_end), step=1, key=f"end_{idx}")
            else:
                start_h, end_h = 0, 0
            return (mode, price_val, discount, start_h, end_h)

    # Plans 1–3 (defaults)
    plans.append(plan_card(cols[2], 1, default_mode="All day", default_discount=6.0))
    plans.append(plan_card(cols[3], 2, default_mode="By hour", default_discount=20.0, default_start=23, default_end=7))
    plans.append(plan_card(cols[4], 3, default_mode="All day", default_discount=0.0))

    # If revealed, render Plan 4 / Plan 5 in the left placeholders
    if st.session_state.plan4_visible:
        plans.append(plan_card(cols[0], 4, default_mode="All day", default_discount=0.0))
    if st.session_state.plan5_visible:
        plans.append(plan_card(cols[1], 5, default_mode="All day", default_discount=0.0))

    st.markdown("---")
    st.subheader("Cost Comparison")

    if not data.empty:
        total_kwh = data["consumption_kwh"].sum()
        base_cost = total_kwh * ref_price
        rows = [("Reference (no discount)", base_cost, 0.0)]
        for i, p in enumerate(plans, start=1):
            c = compute_cost(data, p[1], p[0], p[2], p[3], p[4])
            rows.append((f"Plan {i}", c, base_cost - c))

        df_costs = pd.DataFrame(rows, columns=["Plan", "Total cost (NIS)", "Savings vs ref. (NIS)"])

        # Force display with exactly ONE decimal place
        df_costs["Total cost (NIS)"] = df_costs["Total cost (NIS)"].astype(float)
        df_costs["Savings vs ref. (NIS)"] = df_costs["Savings vs ref. (NIS)"].astype(float)

        # Only pick the cheapest *actual* plan (ignore reference)
        plan_rows = df_costs[df_costs["Plan"].str.startswith("Plan")]
        min_cost = plan_rows["Total cost (NIS)"].min() if not plan_rows.empty else np.inf

        def highlight_best(row):
            if row["Plan"].startswith("Plan") and float(row["Total cost (NIS)"]) == float(min_cost):
                return ['background-color: var(--row-green); color: var(--row-green-text); font-weight: 600'] * len(row)
            else:
                return [''] * len(row)

        styled = (
            df_costs
            .style
            .format({"Total cost (NIS)": "{:.1f}", "Savings vs ref. (NIS)": "{:.1f}"})
            .apply(highlight_best, axis=1)
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("No data to compute costs.")

    st.markdown('</div>', unsafe_allow_html=True)
