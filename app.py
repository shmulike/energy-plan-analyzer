# app.py — Energy Plan Analyzer (robust CSV loader + theme + winner text highlight)
import io, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ===================== Config =====================
DEFAULT_PRICE_KWH   = 64.02  # NIS per kWh (reference row)
DAY_COLOR_DEFAULT   = "#6200EE"   # Day (purple)
NIGHT_COLOR_DEFAULT = "#03DAC6"   # Night (teal)
ALLOW_COLOR_CHANGE  = False       # user color pickers off
MAX_PLANS           = 5
NIGHT_START, NIGHT_END = 23, 7    # night window for split

PLOT_WIDTH_INCHES  = 12
PLOT_HEIGHT_INCHES = 6
AXIS_TEXT_SIZE     = 10
LEGEND_TEXT_SIZE   = 10

# ===================== Page setup =====================
st.set_page_config(page_title="Electricity Consumption Dashboard", layout="wide")

# ===================== Theme state + visible switch =====================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # default dark

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
    if hasattr(st, "toggle"):
        st.session_state.dark_mode = bool(
            st.toggle("🌙 Dark mode", value=st.session_state.dark_mode, key="__dark_toggle__")
        )
    else:
        st.session_state.dark_mode = bool(
            st.checkbox("🌙 Dark mode", value=st.session_state.dark_mode, key="__dark_checkbox__")
        )

# ===================== Theme CSS =====================
if st.session_state.dark_mode:
    BG, CARD, MUTED, BORDER, TEXT = "#0f1115", "#161b22", "#8b949e", "#30363d", "#e6edf3"
    ACCENT, ACCENT_H = "#0A84FF", "#066de6"
    DF_TH_BG, DF_TD_BG, DF_TEXT = CARD, CARD, TEXT
else:
    BG, CARD, MUTED, BORDER, TEXT = "#f5f5f7", "#ffffff", "#6e6e73", "#e5e5ea", "#1d1d1f"
    ACCENT, ACCENT_H = "#0A84FF", "#0066d6"
    DF_TH_BG, DF_TD_BG, DF_TEXT = CARD, CARD, TEXT

st.markdown(
    f"""
<style>
.stApp{{background-color:{BG};color:{TEXT};}}
.block-container{{padding-top:.6rem;padding-bottom:.6rem;}}
.card{{background:{CARD};border-radius:16px;padding:1rem;border:1px solid {BORDER};
      box-shadow:0 8px 24px rgba(0,0,0,.12);}}
h1,h2,h3,h4,h5{{color:{TEXT};margin:0 0 .6rem 0;}}
input,textarea,select,.stTextInput input,.stNumberInput input,
.stSelectbox div[role="button"],.stTextArea textarea{{
  background:transparent!important;color:{TEXT}!important;border:1px solid {BORDER}!important;
  border-radius:12px!important;
}}
.stNumberInput button,.stSelectbox svg{{color:{TEXT}!important;}}
.stButton>button{{background:{ACCENT};color:#fff;border-radius:12px;border:none;
  padding:.45rem .9rem;font-weight:600;box-shadow:0 6px 16px rgba(10,132,255,.25);}}
.stButton>button:hover{{background:{ACCENT_H};}}
.dataframe th{{background:{DF_TH_BG}!important;color:{DF_TEXT}!important;border:1px solid {BORDER}!important;}}
.dataframe td{{background:{DF_TD_BG}!important;color:{DF_TEXT}!important;border:1px solid {BORDER}!important;}}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== Robust CSV loader =====================
HEB_DATE = ["תאריך", "תאריך קריאה", "Date"]
HEB_TIME = ["מועד תחילת הפעימה", "שעה", "Time"]
HEB_KWH  = ['צריכה בקוט"ש', 'צריכה בקוט""ש', "צריכה בקוטש", "צריכה", "kWh", "Consumption (kWh)"]
TS_CAND  = ["timestamp", "Datetime", "DateTime", "תאריך ושעה", "תאריך_שעה"]

def _find_header_line(raw_bytes: bytes) -> int:
    """Find the first line that looks like a header (Hebrew/English date+time)."""
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    for i, ln in enumerate(lines[:300]):
        if any(tok in ln for tok in HEB_DATE) and (any(tok in ln for tok in HEB_TIME) or any(tok in ln for tok in HEB_KWH)):
            return i
    return 0

def load_csv(file) -> pd.DataFrame:
    """Return DataFrame with columns: datetime (UTC+local), kwh (float), hour (int)"""
    if file is None:
        return pd.DataFrame(columns=["datetime", "kwh", "hour"])
    raw = file.read()

    header_line = _find_header_line(raw)

    # Try encodings and flexible sep inference
    for enc in ("utf-8", "cp1255"):
        try:
            df = pd.read_csv(io.BytesIO(raw), skiprows=header_line, encoding=enc, sep=None, engine="python")
            if not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        st.error("Could not parse CSV (encoding/format). Try exporting again or share a sample.")
        return pd.DataFrame(columns=["datetime", "kwh", "hour"])

    # Normalize column names (strip + collapse spaces)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # Try direct timestamp path
    ts_col = next((c for c in df.columns if any(t.lower() == str(c).lower() for t in TS_CAND)), None)
    kwh_col = next((c for c in df.columns if any(k == c for k in HEB_KWH)), None)

    # If not found, try date+time mapping
    if ts_col is None:
        date_col = next((c for c in df.columns if any(d == c for d in HEB_DATE)), None)
        time_col = next((c for c in df.columns if any(t == c for t in HEB_TIME)), None)
        if date_col is not None and time_col is not None:
            df["__timestamp__"] = df[date_col].astype(str) + " " + df[time_col].astype(str)
            ts_col = "__timestamp__"

    # As a last resort, guess numeric column for kWh if not found
    if kwh_col is None:
        numeric_candidates = []
        for c in df.columns:
            try:
                pd.to_numeric(df[c], errors="raise")
                numeric_candidates.append(c)
            except Exception:
                pass
        kwh_col = numeric_candidates[0] if numeric_candidates else None

    if ts_col is None or kwh_col is None:
        st.error("Could not locate timestamp and energy columns in the CSV.")
        return pd.DataFrame(columns=["datetime", "kwh", "hour"])

    # Parse datetime (dayfirst because IEC exports are usually dd/mm/yyyy)
    dt = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
    kwh = pd.to_numeric(df[kwh_col], errors="coerce")

    out = pd.DataFrame({"datetime": dt, "kwh": kwh}).dropna().sort_values("datetime")
    if out.empty:
        st.error("No valid rows after parsing datetime and kWh.")
        return pd.DataFrame(columns=["datetime", "kwh", "hour"])

    out["hour"] = out["datetime"].dt.hour
    return out.reset_index(drop=True)

# ===================== Helpers =====================
def aggregate_week(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly stacked totals: day_kwh, night_kwh by ISO week."""
    if df.empty:
        return pd.DataFrame(columns=["label", "day_kwh", "night_kwh"])
    is_night = (df["hour"] >= NIGHT_START) | (df["hour"] < NIGHT_END)
    work = df.copy()
    work["is_night"] = is_night.astype(int)
    iso = work["datetime"].dt.isocalendar()
    work["label"] = iso.year.astype(str) + "-W" + iso.week.astype(str)
    grp = work.groupby(["label", "is_night"])["kwh"].sum().unstack(fill_value=0)
    grp.columns = ["day_kwh", "night_kwh"]
    return grp.reset_index()

def plot_stacked(df_agg: pd.DataFrame, title: str, day_color: str, night_color: str, dark: bool):
    if df_agg.empty:
        st.warning("No data to plot.")
        return
    labels = df_agg["label"].tolist()
    day = df_agg["day_kwh"].values
    night = df_agg["night_kwh"].values

    fig = plt.figure(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    ax = fig.add_subplot(111)
    if dark:
        fig.patch.set_facecolor("#0f1115"); ax.set_facecolor("#0f1115")
        axis_color = "#e6edf3"; spine_color = "#8b949e"
        legend_face = "#161b22"; legend_edge = "#30363d"; legend_text = "#FFFFFF"
    else:
        fig.patch.set_facecolor("#f5f5f7"); ax.set_facecolor("#FFFFFF")
        axis_color = "#1d1d1f"; spine_color = "#d2d2d7"
        legend_face = "#FFFFFF"; legend_edge = "#e5e5ea"; legend_text = "#1d1d1f"

    for s in ax.spines.values(): s.set_color(spine_color)
    x = np.arange(len(labels))
    ax.bar(x, day,   label="Day",   color=day_color)
    ax.bar(x, night, bottom=day, label="Night", color=night_color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=AXIS_TEXT_SIZE, color=axis_color)
    ax.set_xlabel("Week", fontsize=AXIS_TEXT_SIZE, color=axis_color)
    ax.set_ylabel("Consumption (kWh)", fontsize=AXIS_TEXT_SIZE, color=axis_color)
    ax.set_title(title, fontsize=AXIS_TEXT_SIZE+2, color=axis_color)

    leg = ax.legend(fontsize=LEGEND_TEXT_SIZE, facecolor=legend_face, edgecolor=legend_edge)
    for txt in leg.get_texts(): txt.set_color(legend_text)

    ax.tick_params(colors=axis_color)
    fig.tight_layout()
    st.pyplot(fig)

def compute_cost(df: pd.DataFrame, mode: str, price_kwh: float, discount_pct: float,
                 start_hour: int | None = None, end_hour: int | None = None) -> float:
    if df.empty:
        return 0.0
    price_kwh = float(price_kwh); discount_pct = float(discount_pct)
    total = df["kwh"].sum()

    if mode == "All day":
        return total * price_kwh * (1 - discount_pct/100.0)

    # By hour
    sh = 0 if start_hour is None else int(start_hour)
    eh = 0 if end_hour   is None else int(end_hour)

    if sh == eh:
        mask = np.ones(len(df), dtype=bool)  # whole day
    elif sh < eh:
        mask = (df["hour"] >= sh) & (df["hour"] < eh)
    else:
        # wraps over midnight (e.g., 23→7)
        mask = (df["hour"] >= sh) | (df["hour"] < eh)

    win = df.loc[mask, "kwh"].sum()
    rest = total - win
    return rest * price_kwh + win * price_kwh * (1 - discount_pct/100.0)

# ===================== Session for extra plans =====================
if "num_plans" not in st.session_state:
    st.session_state.num_plans = 3  # show 3 by default

# ===================== TOP: Controls + Chart =====================
left, right = st.columns([4, 8], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Electricity CSV", type=["csv"])

    st.subheader("Aggregation")
    granularity = st.radio("Choose", ["Week"], horizontal=True, index=0)

    # Colors (locked by default)
    if ALLOW_COLOR_CHANGE:
        st.subheader("Colors")
        day_color = st.color_picker("Day bar", DAY_COLOR_DEFAULT, key="c_day")
        night_color = st.color_picker("Night bar", NIGHT_COLOR_DEFAULT, key="c_night")
    else:
        day_color, night_color = DAY_COLOR_DEFAULT, NIGHT_COLOR_DEFAULT

    st.subheader("Reference price")
    ref_price = st.number_input("Electric price per kWh (NIS)", value=DEFAULT_PRICE_KWH, step=0.01, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dashboard")
    data = load_csv(uploaded)
    if not data.empty:
        st.caption(f"Loaded {len(data):,} rows. Range: {data['datetime'].min().date()} → {data['datetime'].max().date()}")
        agg_week = aggregate_week(data)
        plot_stacked(agg_week, "Consumption (Week) - Day vs Night",
                     day_color, night_color, dark=st.session_state.dark_mode)
    else:
        st.info("Upload a CSV to see plot and pricing.")
        agg_week = pd.DataFrame()
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== BOTTOM: Full‑width Plans + Cost Comparison =====================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Pricing Plans")

# Add/Remove plan buttons
btn_add, btn_remove = st.columns(2)
with btn_add:
    if st.button("＋ Add Plan", use_container_width=True):
        if st.session_state.num_plans < MAX_PLANS:
            st.session_state.num_plans += 1
with btn_remove:
    if st.button("– Remove Plan", use_container_width=True):
        if st.session_state.num_plans > 3:
            st.session_state.num_plans -= 1

# Plans 1..N left → right
plans = []
cols = st.columns(st.session_state.num_plans, gap="large")
for i, col in enumerate(cols):
    with col:
        st.markdown(f"**Plan {i+1}**")
        mode = st.selectbox(f"Mode P{i+1}", ["All day", "By hour"], key=f"mode_{i}")
        price_val = st.number_input(f"Price P{i+1} (NIS/kWh)", value=ref_price, step=0.01, format="%.2f", key=f"price_{i}")
        discount = st.number_input(f"Discount % P{i+1}", value=0.0, step=0.1, format="%.1f", key=f"disc_{i}")
        if mode == "By hour":
            c1, c2 = st.columns(2)
            with c1:
                start_h = st.number_input("Start", min_value=0, max_value=23, value=23, step=1, key=f"start_{i}")
            with c2:
                end_h   = st.number_input("End",   min_value=0, max_value=23, value=7,  step=1, key=f"end_{i}")
        else:
            start_h, end_h = None, None
        plans.append((mode, price_val, discount, start_h, end_h))

st.markdown("---")
st.subheader("Cost Comparison")

if not data.empty and len(plans) > 0:
    total_kwh = data["kwh"].sum()
    base_cost = total_kwh * ref_price
    rows = [("Reference (no discount)", base_cost, 0.0)]
    for i, p in enumerate(plans, start=1):
        c = compute_cost(data, *p)
        rows.append((f"Plan {i}", c, base_cost - c))

    df_costs = pd.DataFrame(rows, columns=["Plan", "Total cost (NIS)", "Savings vs ref. (NIS)"])
    df_costs["Total cost (NIS)"]      = df_costs["Total cost (NIS)"].astype(float)
    df_costs["Savings vs ref. (NIS)"] = df_costs["Savings vs ref. (NIS)"].astype(float)

    # Winner detection (tolerant to rounding)
    is_plan = df_costs["Plan"].str.startswith("Plan")
    min_cost = df_costs.loc[is_plan, "Total cost (NIS)"].min() if is_plan.any() else np.inf
    winners = is_plan & np.isclose(df_costs["Total cost (NIS)"], min_cost, rtol=0.0, atol=0.05)

    # Only change text color for winner (keep default background)
    WIN_TEXT = "color: #90EE90 !important; font-weight: 700 !important;"

    def highlight_winner_text(row):
        return [WIN_TEXT if winners.loc[row.name] else "" for _ in row]

    base_styles = [
        {"selector": "th", "props": [("background-color", DF_TH_BG),
                                     ("color", DF_TEXT),
                                     ("border", f"1px solid {BORDER}")]},
        {"selector": "td", "props": [("background-color", DF_TD_BG),
                                     ("color", DF_TEXT),
                                     ("border", f"1px solid {BORDER}")]},
    ]

    styled = (
        df_costs
        .style
        .set_table_styles(base_styles)
        .format({"Total cost (NIS)": "{:.1f}", "Savings vs ref. (NIS)": "{:.1f}"})
        .apply(highlight_winner_text, axis=1)
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.info("No data / plans to compare yet.")

st.markdown('</div>', unsafe_allow_html=True)
