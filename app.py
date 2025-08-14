import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =============================
# CONFIGURATION VARIABLES
# =============================
DEFAULT_PRICE_KWH = 64.02
DEFAULT_DAY_COLOR = "#6200EE"   # Apple-like purple
DEFAULT_NIGHT_COLOR = "#03DAC6" # Apple-like teal
ALLOW_COLOR_CHANGE = False      # User can change colors?
MAX_PLANS = 5                   # Max number of plans allowed

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Electricity Consumption Dashboard", layout="wide")

# =============================
# THEME STATE INIT
# =============================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Top-right toggle for theme
col_a, col_b = st.columns([6, 1])
with col_b:
    try:
        st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
    except Exception:
        st.session_state.dark_mode = st.checkbox("Dark mode", value=st.session_state.dark_mode)

# Theme colors
if st.session_state.dark_mode:
    BG_COLOR = "#1E1E1E"
    TEXT_COLOR = "#FFFFFF"
    DF_TH_BG = "#2C2C2C"
    DF_TD_BG = "#1E1E1E"
    BORDER = "#555555"
else:
    BG_COLOR = "#FFFFFF"
    TEXT_COLOR = "#000000"
    DF_TH_BG = "#F2F2F2"
    DF_TD_BG = "#FFFFFF"
    BORDER = "#DDDDDD"

# Apply background and text color globally
st.markdown(
    f"""
    <style>
    body {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# INTRO TEXT
# =============================
st.markdown("""
### Electricity Consumption Dashboard
This tool, created by **Shmulik Edelman**, helps you upload and analyze your household or business electricity usage.  
Compare up to five custom pricing plans, visualize day vs. night consumption, and see which plan saves you the most money.
""")

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader("Upload your electricity CSV file", type="csv")

# =============================
# PLAN SETUP
# =============================
if "num_plans" not in st.session_state:
    st.session_state.num_plans = 3  # Default 3 visible plans

def add_plan():
    if st.session_state.num_plans < MAX_PLANS:
        st.session_state.num_plans += 1

def remove_plan():
    if st.session_state.num_plans > 3:
        st.session_state.num_plans -= 1

# Buttons for adding/removing plans
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("+ Add Plan"):
        add_plan()
with col_btn2:
    if st.button("- Remove Plan"):
        remove_plan()

# =============================
# PLAN INPUTS
# =============================
plan_inputs = []
plan_cols = st.columns(st.session_state.num_plans)
for i, col in enumerate(plan_cols):
    with col:
        st.markdown(f"**Plan {i+1}**")
        all_day = st.radio(f"Pricing type (Plan {i+1})", ["All day", "By hour"], key=f"ptype_{i}")
        if all_day == "All day":
            price = st.number_input(f"Price/kWh (Plan {i+1})", value=DEFAULT_PRICE_KWH, key=f"price_{i}")
            disc = st.number_input(f"Discount % (Plan {i+1})", value=0.0, step=0.1, key=f"disc_{i}")
            plan_inputs.append(("all_day", price, disc, None, None))
        else:
            price = st.number_input(f"Price/kWh (Plan {i+1})", value=DEFAULT_PRICE_KWH, key=f"price_{i}")
            start_hour = st.number_input(f"Start hour (0-23)", min_value=0, max_value=23, step=1, key=f"start_{i}")
            end_hour = st.number_input(f"End hour (0-23)", min_value=0, max_value=23, step=1, key=f"end_{i}")
            disc = st.number_input(f"Discount % (Plan {i+1})", value=0.0, step=0.1, key=f"disc_{i}")
            plan_inputs.append(("by_hour", price, disc, start_hour, end_hour))

# =============================
# DATA PROCESSING & PLOT
# =============================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Remove empty header row if exists
    if df.columns[0] != "timestamp":
        df.columns = df.iloc[0]
        df = df.drop(0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["kWh"] = df["kWh"].astype(float)
    df["hour"] = df["timestamp"].dt.hour

    # Define night period
    df["is_night"] = (df["hour"] >= 23) | (df["hour"] < 7)

    # Aggregate weekly
    df["week"] = df["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
    agg = df.groupby("week").agg(
        day_kwh=("kWh", lambda x: x[~df.loc[x.index, "is_night"]].sum()),
        night_kwh=("kWh", lambda x: x[df.loc[x.index, "is_night"]].sum())
    ).reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    day_color = DEFAULT_DAY_COLOR
    night_color = DEFAULT_NIGHT_COLOR
    ax.bar(agg["week"], agg["day_kwh"], color=day_color, label="Day usage")
    ax.bar(agg["week"], agg["night_kwh"], bottom=agg["day_kwh"], color=night_color, label="Night usage")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Week")
    ax.tick_params(axis='x', rotation=45)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    st.pyplot(fig)

    # =============================
    # COST COMPARISON
    # =============================
    def compute_cost(df, ptype, price, disc, sh, eh):
        if ptype == "all_day":
            return df["kWh"].sum() * price * (1 - disc / 100)
        else:
            mask = (df["hour"] >= sh) & (df["hour"] < eh)
            kwh_disc = df.loc[mask, "kWh"].sum()
            kwh_rest = df.loc[~mask, "kWh"].sum()
            return kwh_disc * price * (1 - disc / 100) + kwh_rest * price

    costs = []
    for i, plan in enumerate(plan_inputs):
        c = compute_cost(df, *plan)
        costs.append(("Plan " + str(i+1), c, costs[0][1] - c if i > 0 else 0))

    df_costs = pd.DataFrame(costs, columns=["Plan", "Total cost (NIS)", "Savings vs ref. (NIS)"])

    # Winner detection
    is_plan = df_costs["Plan"].str.startswith("Plan")
    min_cost = df_costs.loc[is_plan, "Total cost (NIS)"].min() if is_plan.any() else np.inf
    winners = is_plan & np.isclose(df_costs["Total cost (NIS)"], min_cost, atol=0.05)

    # Only text color change
    WIN_TEXT = "color: #90EE90 !important; font-weight: 700 !important;"

    def highlight_winner_text(row):
        return [WIN_TEXT if winners.loc[row.name] else "" for _ in row]

    base_styles = [
        {"selector": "th", "props": [("background-color", DF_TH_BG),
                                     ("color", TEXT_COLOR),
                                     ("border", f"1px solid {BORDER}")]},
        {"selector": "td", "props": [("background-color", DF_TD_BG),
                                     ("color", TEXT_COLOR),
                                     ("border", f"1px solid {BORDER}")]},
    ]

    styled = (
        df_costs
        .style
        .set_table_styles(base_styles)
        .format({"Total cost (NIS)": "{:.1f}", "Savings vs ref. (NIS)": "{:.1f}"})
        .apply(highlight_winner_text, axis=1)
    )

    st.subheader("Cost Comparison")
    st.dataframe(styled, use_container_width=True)
