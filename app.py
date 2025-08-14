# streamlit_power_dashboard_v2.py
# Streamlit dashboard for electricity consumption CSV uploads
# Features you asked for:
# - Left panel: CSV upload, Day/Week/Month switch (radio as a "switch"), color toggles with default buttons,
#               text box for electric price per kWh (default 64.02 agorot = 0.6402 NIS)
# - Right panel: stacked bar plot (Day bottom, Night top), night = 23:00–07:00
# - Variables for defaults: electric price, plot width/height, axis text size, legend text size
# - Pricing Plans 1/2/3 section below the plot:
#   * Switch between "All day" and "By hour"
#   * If All day: price + discount (% with 1 decimal)
#   * If By hour: start hour + end hour + price + discount
# - Robust CSV parsing for Israeli "LP" export with Hebrew headers and preamble

import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =============================
# Defaults (easy to tweak)
# =============================
DEFAULT_ELECTRIC_PRICE = 0.6402  # NIS per kWh (64.02 agorot)
PLOT_WIDTH_INCHES = 12
PLOT_HEIGHT_INCHES = 6
AXIS_TEXT_SIZE = 10
LEGEND_TEXT_SIZE = 10

# Fixed night definition for the plot
NIGHT_START = 23
NIGHT_END = 7  # exclusive

st.set_page_config(page_title="Electricity Consumption Dashboard", layout="wide")


# =============================
# Helpers
# =============================
def find_header_line(raw_bytes: bytes) -> int:
    """
    For files with a preamble, find the 0-based line index of the header row
    that includes the expected Hebrew column names.
    Returns 0 if not found.
    """
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
    """
    Load the CSV into canonical dataframe with columns:
    ['datetime', 'consumption_kwh']
    Handles localized (Hebrew) headers: 'תאריך','מועד תחילת הפעימה','צריכה בקוט"ש'
    """
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

    # Normalize column names (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    # Heuristics to find date/time/consumption columns
    col_map = {"date": None, "time": None, "cons": None}

    # Common Hebrew / English headers
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
            if col_map[key] is not None:
                break

    # Guess by content if still missing
    if col_map["date"] is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(5).tolist())
            if "/" in sample or "-" in sample:
                col_map["date"] = c
                break
    if col_map["time"] is None:
        for c in df.columns:
            sample = str(df[c].dropna().astype(str).head(5).tolist())
            if ":" in sample:
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
            means = [(c, pd.to_numeric(df[c], errors="coerce").mean()) for c in num_cols]
            col_map["cons"] = sorted(means, key=lambda x: (x[1] if x[1] is not None else -np.inf), reverse=True)[0][0]

    if col_map["date"] is None or col_map["time"] is None or col_map["cons"] is None:
        return pd.DataFrame(columns=["datetime", "consumption_kwh"])

    cons = pd.to_numeric(df[col_map["cons"]], errors="coerce")
    dt = pd.to_datetime(
        df[col_map["date"]].astype(str) + " " + df[col_map["time"]].astype(str),
        dayfirst=True, errors="coerce"
    )
    out = pd.DataFrame({"datetime": dt, "consumption_kwh": cons})
    out = out.dropna(subset=["datetime", "consumption_kwh"]).sort_values("datetime").reset_index(drop=True)
    return out


def split_day_night(df: pd.DataFrame, start_hour=NIGHT_START, end_hour=NIGHT_END) -> pd.Series:
    hrs = df["datetime"].dt.hour
    return (hrs >= start_hour) | (hrs < end_hour)


def aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    granularity: 'Day', 'Week', or 'Month'
    Returns columns: ['label','day_kwh','night_kwh']
    """
    if df.empty:
        return pd.DataFrame(columns=["label", "day_kwh", "night_kwh"])

    is_night = split_day_night(df)
    work = df.copy()
    work["is_night"] = is_night.astype(int)

    if granularity == "Day":
        work["period"] = work["datetime"].dt.date.astype(str)
    elif granularity == "Week":
        work["year"] = work["datetime"].dt.isocalendar().year
        work["week"] = work["datetime"].dt.isocalendar().week
        work["period"] = work["year"].astype(str) + "-W" + work["week"].astype(str)
    else:  # Month
        work["period"] = work["datetime"].dt.to_period("M").astype(str)

    grp = work.groupby(["period", "is_night"])["consumption_kwh"].sum().unstack(fill_value=0)
    grp.columns = ["day_kwh", "night_kwh"]  # 0=day,1=night
    grp = grp.reset_index().rename(columns={"period": "label"})
    return grp


def plot_stacked(df_agg: pd.DataFrame, title: str, color_day: str, color_night: str):
    if df_agg.empty:
        st.warning("No data to plot.")
        return

    labels = df_agg["label"].tolist()
    day = df_agg["day_kwh"].values
    night = df_agg["night_kwh"].values

    fig = plt.figure(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    x = np.arange(len(labels))
    plt.bar(x, day, label="Day", linewidth=0, color=color_day)
    plt.bar(x, night, bottom=day, label="Night", linewidth=0, color=color_night)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=AXIS_TEXT_SIZE)
    plt.xlabel("Period", fontsize=AXIS_TEXT_SIZE)
    plt.ylabel("Consumption (kWh)", fontsize=AXIS_TEXT_SIZE)
    plt.title(title, fontsize=AXIS_TEXT_SIZE + 2)
    plt.legend(fontsize=LEGEND_TEXT_SIZE)
    plt.tight_layout()
    st.pyplot(fig)


def compute_cost(df: pd.DataFrame, price_kwh: float, mode: str, discount_pct: float,
                 start_hour: int = 0, end_hour: int = 0) -> float:
    """
    mode: 'All day' or 'By hour'
    discount_pct: percentage (e.g., 6.0 means 6%)
    For 'By hour', discount applies only to [start_hour, end_hour) with wrap at midnight.
    Returns total cost in NIS.
    """
    if df.empty:
        return 0.0
    cons = df.copy()
    cons["hour"] = cons["datetime"].dt.hour
    total = cons["consumption_kwh"].sum()

    if mode == "All day":
        cost = total * price_kwh * (1 - discount_pct / 100.0)
    else:
        # By hour window with wrap
        if start_hour == end_hour:
            window_mask = np.ones(len(cons), dtype=bool)  # full-day window
        elif start_hour < end_hour:
            window_mask = (cons["hour"] >= start_hour) & (cons["hour"] < end_hour)
        else:
            # wraps midnight
            window_mask = (cons["hour"] >= start_hour) | (cons["hour"] < end_hour)

        cons_window = cons.loc[window_mask, "consumption_kwh"].sum()
        cons_rest = total - cons_window
        cost = cons_rest * price_kwh + cons_window * price_kwh * (1 - discount_pct / 100.0)
    return float(cost)


# =============================
# UI Layout
# =============================
left, right = st.columns([4, 8], gap="large")

with left:
    st.header("Controls")

    # Day/Week/Month "switch" (three-way). Using radio as a clean switch.
    granularity = st.radio("Aggregation", options=["Day", "Week", "Month"], horizontal=True, index=2)

    # Colors with a "default" toggle for each (like a bar with a toggle & default button)
    st.subheader("Colors")
    default_day = st.toggle("Use default Day color (skyblue)", value=True)
    day_color = "#87CEEB"  # skyblue
    if not default_day:
        day_color = st.color_picker("Pick Day bar color", value=day_color, key="day_color")

    default_night = st.toggle("Use default Night color (navy)", value=True)
    night_color = "#000080"  # navy
    if not default_night:
        night_color = st.color_picker("Pick Night bar color", value=night_color, key="night_color")

    # Price input (global reference price)
    st.subheader("Reference price")
    ref_price = st.text_input("Electric price per kWh (NIS)", value=f"{DEFAULT_ELECTRIC_PRICE:.4f}")
    try:
        ref_price = float(ref_price)
    except ValueError:
        st.warning("Invalid price; using default.")
        ref_price = DEFAULT_ELECTRIC_PRICE

    # File upload
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Upload electricity CSV", type=["csv"])

    # Optional notes
    notes = st.text_area("Notes (optional)")

with right:
    st.header("Dashboard")

    data = load_csv(uploaded)
    if not data.empty:
        st.caption(f"Loaded {len(data):,} rows. Range: {data['datetime'].min().date()} → {data['datetime'].max().date()}")
    else:
        st.info("Upload a CSV to render the plot and pricing comparisons.")

    agg_df = aggregate(data, granularity=granularity)
    plot_stacked(agg_df, title=f"Consumption ({granularity}) - Day vs Night",
                 color_day=day_color, color_night=night_color)

    st.markdown("---")
    st.subheader("Pricing Plans 1, 2, 3")

    # Plans configuration block
    def plan_block(idx: int, default_price: float):
        st.markdown(f"**Plan {idx}**")
        cols = st.columns([2, 2, 2, 2, 2])
        with cols[0]:
            mode = st.selectbox(f"Mode (Plan {idx})", ["All day", "By hour"], key=f"mode_{idx}")
        with cols[1]:
            price_text = st.text_input(f"Price (NIS/kWh) P{idx}", value=f"{default_price:.4f}", key=f"price_{idx}")
            try:
                price_val = float(price_text)
            except ValueError:
                price_val = default_price
        with cols[2]:
            discount = st.number_input(
                f"Discount % P{idx}",
                min_value=0.0, max_value=100.0,
                value=6.0 if idx == 1 else (20.0 if idx == 2 else 0.0),
                step=0.1, format="%.1f", key=f"disc_{idx}"
            )
        with cols[3]:
            start_h = st.number_input(f"Start hour P{idx}", min_value=0, max_value=23,
                                      value=23 if idx == 2 else 0, step=1, key=f"start_{idx}")
        with cols[4]:
            end_h = st.number_input(f"End hour P{idx}", min_value=0, max_value=23,
                                    value=7 if idx == 2 else 0, step=1, key=f"end_{idx}")

        if mode == "All day":
            st.caption("Discount applies to the entire day.")
        else:
            st.caption("Discount applies only within [start hour, end hour) (wraps midnight if start > end).")
        return mode, price_val, discount, start_h, end_h

    p1 = plan_block(1, ref_price)
    p2 = plan_block(2, ref_price)
    p3 = plan_block(3, ref_price)

    # Calculate and display cost table
    st.markdown("#### Cost Comparison")
    if data.empty:
        st.info("Upload a CSV to compute plan costs.")
    else:
        total_kwh = data["consumption_kwh"].sum()
        base_cost = total_kwh * ref_price

        def cost_for(plan):
            mode, price_val, disc, sh, eh = plan
            return compute_cost(data, price_kwh=price_val, mode=mode,
                                discount_pct=disc, start_hour=sh, end_hour=eh)

        c1 = cost_for(p1)
        c2 = cost_for(p2)
        c3 = cost_for(p3)

        table = pd.DataFrame({
            "Plan": ["No discount (ref.)", "Plan 1", "Plan 2", "Plan 3"],
            "Total cost (NIS)": [base_cost, c1, c2, c3],
            "Savings vs ref. (NIS)": [0.0, base_cost - c1, base_cost - c2, base_cost - c3]
        })
        table["Total cost (NIS)"] = table["Total cost (NIS)"].round(1)
        table["Savings vs ref. (NIS)"] = table["Savings vs ref. (NIS)"].round(1)
        st.dataframe(table, use_container_width=True)

    if notes.strip():
        st.markdown("---")
        st.subheader("Notes")
        st.write(notes.strip())

st.caption("Plot night period is fixed at 23:00–07:00. 'By hour' plan window wraps midnight if start > end.")
