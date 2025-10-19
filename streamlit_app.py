# streamlit_app.py - Google Sheets only (secrets-based), robust chart handling + totals + soft-delete + refresh
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Spend", layout="wide")
st.title("💳 Daily Spending")

# ------------------ Modules ------------------
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import. Add transform.py to the same directory.")
    st.exception(e)
    st.stop()

try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

try:
    charts_mod = importlib.import_module("charts")
except Exception:
    charts_mod = None

# ------------------ Config (from Streamlit secrets) ------------------
if not hasattr(st, "secrets") or not st.secrets:
    st.error("Streamlit secrets are not configured. Add SHEET_ID, RANGE, APPEND_RANGE and gcp_service_account to Streamlit secrets.")
    st.stop()

SHEET_ID = st.secrets.get("SHEET_ID")
RANGE = st.secrets.get("RANGE")
APPEND_RANGE = st.secrets.get("APPEND_RANGE")
if not SHEET_ID or not RANGE or not APPEND_RANGE:
    st.error("Missing required secrets: SHEET_ID, RANGE, or APPEND_RANGE. Please add them to Streamlit secrets.")
    st.stop()

# ------------------ Session reload key ------------------
if "reload_key" not in st.session_state:
    st.session_state.reload_key = 0
if "last_refreshed" not in st.session_state:
    st.session_state.last_refreshed = None

# ------------------ Refresh button ------------------
col_refresh, col_time = st.columns([1, 3])
with col_refresh:
    if st.button("🔁 Refresh Data", use_container_width=True):
        st.session_state.reload_key += 1
        st.experimental_rerun()

with col_time:
    if st.session_state.last_refreshed:
        st.caption(f"Last refreshed at: {st.session_state.last_refreshed.strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ------------------ Helpers ------------------
def _get_creds_info():
    if io_mod is None:
        return None
    try:
        if "gcp_service_account" in st.secrets:
            return io_mod.parse_service_account_secret(st.secrets["gcp_service_account"])
    except Exception:
        return None
    return None


def _read_sheet_with_index(spreadsheet_id: str, range_name: str, source_name: str, creds_info):
    if io_mod is None:
        return pd.DataFrame()
    try:
        df = io_mod.read_google_sheet(spreadsheet_id, range_name, creds_info=creds_info, creds_file=None)
    except Exception:
        return pd.DataFrame()
    if df is None:
        return pd.DataFrame()
    df = df.reset_index(drop=True)
    if not df.empty:
        df['_sheet_row_idx'] = df.index.astype(int)
    df['_source_sheet'] = source_name
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sheets(spreadsheet_id, range_hist, range_append, creds_info, reload_key):
    """Cached sheet fetch (reloads when reload_key changes)."""
    hist_df = _read_sheet_with_index(spreadsheet_id, range_hist, "history", creds_info)
    app_df = _read_sheet_with_index(spreadsheet_id, range_append, "append", creds_info)
    return hist_df, app_df


# ------------------ Read Google Sheets ------------------
if io_mod is None:
    st.error("io_helpers.py is required for Google Sheets interaction. Add io_helpers.py to the project.")
    st.stop()

creds_info = _get_creds_info()
if creds_info is None:
    st.error("Google service account credentials not found. Store them in Streamlit secrets as 'gcp_service_account'.")
    st.stop()

with st.spinner("Fetching Google Sheets..."):
    history_df, append_df = fetch_sheets(SHEET_ID, RANGE, APPEND_RANGE, creds_info, st.session_state.reload_key)
    st.session_state.last_refreshed = datetime.utcnow()

history_df = history_df if history_df is not None else pd.DataFrame()
append_df = append_df if append_df is not None else pd.DataFrame()

if history_df.empty and append_df.empty:
    st.error("No rows found in configured Google Sheets. Verify sheet names and service account permissions.")
    st.stop()

sheet_full_df = pd.concat([history_df, append_df], ignore_index=True, sort=False)
if '_sheet_row_idx' not in sheet_full_df.columns:
    sheet_full_df['_sheet_row_idx'] = pd.NA

# ------------------ Handle soft-deleted rows ------------------
if 'is_deleted' in sheet_full_df.columns:
    deleted_mask = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
    df_raw = sheet_full_df.loc[~deleted_mask].copy().reset_index(drop=True)
else:
    df_raw = sheet_full_df.copy().reset_index(drop=True)

if df_raw.empty:
    st.warning("No visible data (all rows deleted or hidden). Try refreshing data.")
    st.stop()

# ------------------ Transform + bank detection ------------------
with st.spinner("Cleaning and deriving columns..."):
    converted_df = transform.convert_columns_and_derives(df_raw)

def add_bank_column(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    df = df.copy()
    if 'Bank' in df.columns and not overwrite:
        df['Bank'] = df['Bank'].astype(str).fillna('Unknown')
        return df
    cand_cols = ['bank', 'account', 'description', 'message', 'narration', 'merchant']
    bank_map = {'hdfc': 'HDFC Bank', 'indian bank': 'Indian Bank'}
    df['Bank'] = df[cand_cols].astype(str).agg(' '.join, axis=1).str.lower().map(lambda x: next((v for k, v in bank_map.items() if k in x), 'Unknown'))
    return df

converted_df = add_bank_column(converted_df)

# ------------------ Sidebar Filters ------------------
st.sidebar.header("Filters")
banks_detected = sorted([b for b in converted_df['Bank'].unique() if pd.notna(b)])
sel_banks = st.sidebar.multiselect("Banks", options=banks_detected, default=banks_detected)

converted_df_filtered = converted_df[converted_df['Bank'].isin(sel_banks)] if sel_banks else converted_df

# ------------------ Aggregation ------------------
with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(converted_df_filtered)

month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}

# ------------------ Date Filters ------------------
if not merged.empty:
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    years = sorted(merged['Date'].dt.year.unique().tolist())
    sel_year = st.sidebar.selectbox("Year", ['All'] + [str(y) for y in years], index=0)
    month_frame = merged if sel_year == 'All' else merged[merged['Date'].dt.year == int(sel_year)]
    month_nums = sorted(month_frame['Date'].dt.month.unique().tolist())
    month_choices = [month_map[m] for m in month_nums]
    sel_months = st.sidebar.multiselect("Month(s)", options=month_choices, default=month_choices)
else:
    sel_year, sel_months = 'All', []

# ------------------ Date Range ------------------
tmp = converted_df_filtered.copy()
tmp['timestamp'] = pd.to_datetime(tmp.get('timestamp', tmp.get('date')), errors='coerce')
valid_dates = tmp['timestamp'].dropna()
min_date, max_date = valid_dates.min().date(), valid_dates.max().date()

totals_mode = st.sidebar.radio("Totals mode", ["Single date", "Date range"], index=0)
today = datetime.utcnow().date()
default_date = max(min_date, min(today, max_date))

if totals_mode == "Single date":
    sel_date = st.sidebar.date_input("Pick date", value=default_date, min_value=min_date, max_value=max_date)
    selected_range = (sel_date, sel_date)
else:
    dr = st.sidebar.date_input("Pick start & end", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    selected_range = (dr[0], dr[1]) if isinstance(dr, (tuple, list)) else (dr, dr)

start_sel, end_sel = [d.date() if isinstance(d, datetime) else d for d in selected_range]

# ------------------ Totals ------------------
try:
    tmp_rows = converted_df_filtered.copy()
    tmp_rows['timestamp'] = pd.to_datetime(tmp_rows.get('timestamp', tmp_rows.get('date')), errors='coerce')
    mask = tmp_rows['timestamp'].dt.date.between(start_sel, end_sel)
    sel_df = tmp_rows.loc[mask].copy()

    amt_col = next((c for c in sel_df.columns if c.lower() == 'amount'), None)
    type_col = next((c for c in sel_df.columns if c.lower() == 'type'), None)

    credit_sum = debit_sum = 0.0
    credit_count = debit_count = 0
    if not sel_df.empty and amt_col:
        sel_df['amt'] = pd.to_numeric(sel_df[amt_col], errors='coerce').fillna(0.0)
        if type_col:
            sel_df['type_norm'] = sel_df[type_col].astype(str).str.lower().str.strip()
            credit_mask = sel_df['type_norm'] == 'credit'
            debit_mask = sel_df['type_norm'] == 'debit'
            credit_sum = sel_df.loc[credit_mask, 'amt'].sum()
            debit_sum = sel_df.loc[debit_mask, 'amt'].sum()
            credit_count, debit_count = int(credit_mask.sum()), int(debit_mask.sum())

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Credits ({start_sel} → {end_sel})", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    c2.metric(f"Debits ({start_sel} → {end_sel})", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    c3.metric("Net (Credits − Debits)", f"₹{(credit_sum - debit_sum):,.0f}")
except Exception as e:
    st.error(f"Failed totals: {e}")

# ------------------ Chart ------------------
st.subheader("Daily Spend and Credit")
plot_df = merged.copy()
if sel_year != 'All':
    plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]
if sel_months:
    inv_map = {v: k for k, v in month_map.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]

plot_df = plot_df.sort_values('Date')
if not plot_df.empty:
    show_debit = st.sidebar.checkbox("Show Debit (Total_Spent)", True)
    show_credit = st.sidebar.checkbox("Show Credit (Total_Credit)", True)
    series = [c for c in ['Total_Spent', 'Total_Credit'] if c in plot_df.columns]
    if show_debit and 'Total_Spent' not in series: series.append('Total_Spent')
    if show_credit and 'Total_Credit' not in series: series.append('Total_Credit')
    if charts_mod:
        try:
            charts_mod.render_chart(plot_df, converted_df_filtered, "Daily line", series, 5)
        except Exception:
            st.line_chart(plot_df.set_index('Date')[series])
    else:
        st.line_chart(plot_df.set_index('Date')[series])
else:
    st.info("No data to plot for selected filters.")

# ------------------ Rows table, Soft-delete, Add new row ------------------
# (identical logic as your previous working app)
# Add this in the success blocks:
# st.session_state.reload_key += 1
# st.experimental_rerun()
