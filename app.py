# streamlit_power_dashboard_multi_plan.py

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
MAX_PLANS = 5  # max number of plan columns

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Electricity Consumption Dashboard", layout="wide")

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
                if n == c or re.sub(r'\s+', '', n) == re.sub(r'\s+', '', c):
                    col_map[key] = c
                    break
            if col_map[key]: break
    if col_map["date"] is None:
        for c in df.columns:
            if "/" in str(df[c].iloc[0]) or "-" in str(df[c].iloc[0]):
                col_map["date"] = c
                break
    if col_map["time"] is None:
        for c in df.columns:
            if ":" in str(df[c].iloc[0]):
                col_map["time"] = c
                break
    if col_map["cons"] is None:
        num_cols = []
        for c in df.columns:
            try:
                pd.to_numeric(df[c], errors="raise")
                num_cols.append(c)
            except Exception:
                pass
        if num_cols:
            col_map["cons"] = num_cols[0]
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
    x = np.arange(len(labels))
    plt.bar(x, day, label="Day", color=color_day)
    plt.bar(x, night, bottom=day, label="Night", color=color_night)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=AXIS_TEXT_SIZE)
    plt.xlabel("Period", fontsize=AXIS_TEXT_SIZE)
    plt.ylabel("Consumption (kWh)", fontsize=AXIS_TEXT_SIZE)
    plt.title(title, fontsize=AXIS_TEXT_SIZE+2)
    plt.legend(fontsize=LEGEND_TEXT_SIZE)
    plt.tight_layout()
    st.pyplot(fig)

def compute_cost(df: pd.DataFrame, price_kwh: float, mode: str, discount_pct: float,
                 start_hour: int = 0, end_hour: int = 0) -> float:
    if df.empty:
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
# Layout
# =============================
left, right = st.columns([4, 8], gap="large")

with left:
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

    st.subheader("Number of plans")
    num_plans = st.slider("Select number of plans", min_value=3, max_value=MAX_PLANS, value=3, step=1)

with right:
    st.header("Dashboard")
    data = load_csv(uploaded)
    if not data.empty:
        st.caption(f"Loaded {len(data):,} rows. Range: {data['datetime'].min().date()} → {data['datetime'].max().date()}")
    else:
        st.info("Upload a CSV to see plot and pricing.")
    agg_df = aggregate(data, granularity)
    plot_stacked(agg_df, f"Consumption ({granularity}) - Day vs Night", day_color, night_color)

    st.markdown("---")
    st.subheader("Pricing Plans")
    cols = st.columns(num_plans)
    plans = []
    for idx in range(num_plans):
        with cols[idx]:
            st.markdown(f"**Plan {idx+1}**")
            mode = st.selectbox(f"Mode P{idx+1}", ["All day", "By hour"], key=f"mode_{idx}")
            price_val = st.number_input(f"Price P{idx+1} (NIS/kWh)", value=ref_price, step=0.01, format="%.4f", key=f"price_{idx}")
            discount = st.number_input(f"Discount % P{idx+1}", value=0.0, step=0.1, format="%.1f", key=f"disc_{idx}")
            if mode == "By hour":
                start_h = st.number_input(f"Start hour P{idx+1}", min_value=0, max_value=23, value=0, step=1, key=f"start_{idx}")
                end_h = st.number_input(f"End hour P{idx+1}", min_value=0, max_value=23, value=0, step=1, key=f"end_{idx}")
            else:
                start_h, end_h = 0, 0
            plans.append((mode, price_val, discount, start_h, end_h))

    st.markdown("#### Cost Comparison")
    if not data.empty:
        total_kwh = data["consumption_kwh"].sum()
        base_cost = total_kwh * ref_price
        costs = []
        costs.append(("Reference (no discount)", base_cost, 0.0))
        for idx, plan in enumerate(plans):
            c = compute_cost(data, *plan)
            costs.append((f"Plan {idx+1}", c, base_cost - c))
        df_costs = pd.DataFrame(costs, columns=["Plan", "Total cost (NIS)", "Savings vs ref. (NIS)"])
        df_costs["Total cost (NIS)"] = df_costs["Total cost (NIS)"].round(1)
        df_costs["Savings vs ref. (NIS)"] = df_costs["Savings vs ref. (NIS)"].round(1)

        # Highlight best cost
        min_cost = df_costs["Total cost (NIS)"].min()
        def highlight_best(row):
            if row["Total cost (NIS)"] == min_cost:
                return ['background-color: #c6efce; font-weight: bold'] * len(row)
            else:
                return [''] * len(row)
        st.dataframe(df_costs.style.apply(highlight_best, axis=1), use_container_width=True)
    else:
        st.info("No data to compute costs.")
