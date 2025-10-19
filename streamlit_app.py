# streamlit_app.py - Google Sheets only, hard-coded sheet config (public-safe)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Spend", layout="wide")
st.title("💳 Daily Spending (Google Sheets)")

# ------------------ Modules ------------------
# transform.py must exist and provide convert_columns_and_derives + compute_daily_totals
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import. Add transform.py to the same directory.")
    st.exception(e)
    st.stop()

# io_helpers required for Google Sheets I/O
try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

# charts module (optional)
try:
    charts_mod = importlib.import_module("charts")
except Exception:
    charts_mod = None

# ------------------ Hard-coded Google Sheet configuration ------------------
SHEET_ID = "1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk"
RANGE = "Mock_data"                 # history sheet name/range (read)
APPEND_RANGE = "Mock_manual_addition"  # append sheet (for manual additions)

# ------------------ Helpers ------------------
def _get_creds_info():
    """Return plain creds dict or None (safe to pass into io_helpers functions)."""
    if io_mod is None:
        return None
    try:
        if hasattr(st, "secrets") and st.secrets and "gcp_service_account" in st.secrets:
            return io_mod.parse_service_account_secret(st.secrets["gcp_service_account"])
    except Exception:
        return None
    return None

def _read_sheet_with_index(spreadsheet_id: str, range_name: str, source_name: str, creds_info):
    """Read a sheet and return DataFrame with _sheet_row_idx and _source_sheet columns."""
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

# ------------------ Read data from Google Sheets (hard-coded) ------------------
if io_mod is None:
    st.error("io_helpers.py is required for Google Sheets interaction. Add io_helpers.py to the project.")
    st.stop()

creds_info = _get_creds_info()
if creds_info is None:
    st.error("Google service account credentials not found. Store them in Streamlit secrets as 'gcp_service_account'.")
    st.stop()

with st.spinner("Fetching Google Sheets..."):
    history_df = _read_sheet_with_index(SHEET_ID, RANGE, "history", creds_info)
    append_df = _read_sheet_with_index(SHEET_ID, APPEND_RANGE, "append", creds_info)

# combine
if history_df is None:
    history_df = pd.DataFrame()
if append_df is None:
    append_df = pd.DataFrame()

if history_df.empty and append_df.empty:
    st.error("No rows found in the configured Google Sheets. Check the sheet names and permissions.")
    st.stop()

sheet_full_df = pd.concat([history_df, append_df], ignore_index=True, sort=False)
if '_sheet_row_idx' not in sheet_full_df.columns:
    sheet_full_df['_sheet_row_idx'] = pd.NA

# filter soft-deleted rows if applicable
if 'is_deleted' in sheet_full_df.columns:
    try:
        deleted_mask = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
    except Exception:
        deleted_mask = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
    df_raw = sheet_full_df.loc[~deleted_mask].copy().reset_index(drop=True)
else:
    df_raw = sheet_full_df.copy().reset_index(drop=True)

if df_raw is None or df_raw.empty:
    st.error("No visible data after filtering deleted rows.")
    st.stop()

# ------------------ Transform and bank detection ------------------
with st.spinner("Cleaning and deriving columns..."):
    converted_df = transform.convert_columns_and_derives(df_raw)

def add_bank_column(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    df = df.copy()
    if 'Bank' in df.columns and not overwrite:
        df['Bank'] = df['Bank'].astype(str).where(df['Bank'].notna(), None)
        df['Bank'] = df['Bank'].fillna('Unknown')
        return df

    cand_cols = ['bank', 'account', 'account_name', 'description', 'message', 'narration', 'merchant', 'beneficiary', 'note']
    def _row_text(row):
        parts = []
        for c in cand_cols:
            if c in row.index and pd.notna(row[c]):
                parts.append(str(row[c]))
        return " ".join(parts).lower()

    bank_map = {
        'hdfc': 'HDFC Bank',
        'indian bank': 'Indian Bank',
    }
    try:
        combined = df.apply(_row_text, axis=1)
    except Exception:
        combined = pd.Series([''] * len(df), index=df.index)

    detected = []
    for text in combined:
        found = None
        for patt, name in bank_map.items():
            if patt in text:
                found = name
                break
        detected.append(found if found is not None else None)

    df['Bank'] = detected
    df['Bank'] = df['Bank'].fillna('Unknown')
    return df

converted_df = add_bank_column(converted_df, overwrite=False)

# ------------------ Filters (non-sensitive UI only) ------------------
# Allow user to filter by detected banks (safe)
st.sidebar.header("Filters")
banks_detected = sorted([b for b in converted_df['Bank'].unique() if pd.notna(b)])
sel_banks = st.sidebar.multiselect("Banks", options=banks_detected, default=banks_detected)

if sel_banks:
    converted_df_filtered = converted_df[converted_df['Bank'].isin(sel_banks)].copy()
else:
    converted_df_filtered = converted_df.copy()

# ------------------ Compute daily totals ------------------
with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(converted_df_filtered)

# ------------------ Date range selection (safe clamp) ------------------
# compute min/max from filtered converted_df
try:
    tmp = converted_df_filtered.copy()
    if 'timestamp' in tmp.columns:
        tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], errors='coerce')
    elif 'date' in tmp.columns:
        tmp['timestamp'] = pd.to_datetime(tmp['date'], errors='coerce')
    else:
        tmp['timestamp'] = pd.NaT
    valid_dates = tmp['timestamp'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
    else:
        st.error("No valid dates found in the data (timestamp/date).")
        st.stop()
except Exception as e:
    st.error(f"Failed to determine date range from data: {e}")
    st.stop()

st.sidebar.write(f"Data source: **Google Sheet (configured)**")
st.sidebar.write(f"History sheet: `{RANGE}`   Append sheet: `{APPEND_RANGE}`")

totals_mode = st.sidebar.radio("Totals mode", ["Single date", "Date range"], index=0)
today = datetime.utcnow().date()
default_date = max(min_date, min(today, max_date))

if totals_mode == "Single date":
    selected_date = st.sidebar.date_input("Pick date", value=default_date, min_value=min_date, max_value=max_date)
    selected_date_range_for_totals = (selected_date, selected_date)
else:
    default_range = (min_date, max_date)
    dr = st.sidebar.date_input("Pick start & end", value=default_range, min_value=min_date, max_value=max_date)
    if isinstance(dr, (tuple, list)):
        selected_date_range_for_totals = (dr[0], dr[1])
    else:
        selected_date_range_for_totals = (dr, dr)

# ------------------ Plotting & charts ------------------
plot_df = merged.copy()
if plot_df.empty:
    st.info("No aggregated data available.")
else:
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    st.subheader("Daily Spend and Credit")
    show_debit = st.sidebar.checkbox("Show Debit (Total_Spent)", value=True)
    show_credit = st.sidebar.checkbox("Show Credit (Total_Credit)", value=True)
    series_selected = []
    if show_debit: series_selected.append('Total_Spent')
    if show_credit: series_selected.append('Total_Credit')

    if charts_mod is not None:
        charts_mod.render_chart(plot_df=plot_df, converted_df=converted_df_filtered, chart_type="Daily line", series_selected=series_selected, top_n=5)
    else:
        try:
            st.line_chart(plot_df.set_index('Date')[series_selected] if series_selected else plot_df.set_index('Date')[["Total_Spent","Total_Credit"]])
        except Exception:
            st.line_chart(plot_df.set_index('Date')[["Total_Spent","Total_Credit"]])

# ------------------ Rows view & download ------------------
st.subheader("Rows (matching selection)")
rows_df = converted_df_filtered.copy()
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
else:
    if 'date' in rows_df.columns:
        rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
    else:
        rows_df['timestamp'] = pd.NaT

start_sel, end_sel = selected_date_range_for_totals
if isinstance(start_sel, datetime):
    start_sel = start_sel.date()
if isinstance(end_sel, datetime):
    end_sel = end_sel.date()

if start_sel and end_sel:
    rows_df = rows_df[(rows_df['timestamp'].dt.date >= start_sel) & (rows_df['timestamp'].dt.date <= end_sel)]

# display preferred columns if present
_desired = ['timestamp', 'bank', 'type', 'amount', 'suspicious', 'message']
col_map = {c.lower(): c for c in rows_df.columns}
display_cols = [col_map[d] for d in _desired if d in col_map]

if not display_cols:
    st.warning("Preferred columns not found — showing full table.")
    st.dataframe(rows_df.reset_index(drop=True), use_container_width=True, height=400)
    st.download_button("Download rows (CSV)", rows_df.to_csv(index=False).encode("utf-8"), file_name="transactions_rows.csv", mime="text/csv")
else:
    display_df = rows_df[display_cols].copy()
    for c in display_df.columns:
        lc = c.lower()
        if lc == 'timestamp' or lc == 'date' or lc.startswith('date'):
            display_df[c] = pd.to_datetime(display_df[c], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
        if lc == 'amount':
            display_df[c] = pd.to_numeric(display_df[c], errors='coerce')
    # pretty rename
    pretty_rename = {}
    for c in display_df.columns:
        lc = c.lower()
        if lc == 'timestamp' or lc == 'date' or lc.startswith('date'):
            pretty_rename[c] = 'Timestamp'
        elif lc == 'bank':
            pretty_rename[c] = 'Bank'
        elif lc == 'type':
            pretty_rename[c] = 'Type'
        elif lc == 'amount':
            pretty_rename[c] = 'Amount'
        elif lc == 'suspicious':
            pretty_rename[c] = 'Suspicious'
        elif lc == 'message':
            pretty_rename[c] = 'Message'
    if pretty_rename:
        display_df = display_df.rename(columns=pretty_rename)
    final_order = [c for c in ['Timestamp','Bank','Type','Amount','Suspicious','Message'] if c in display_df.columns]
    display_df = display_df[final_order]
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=420)
    st.download_button("Download rows (CSV)", display_df.to_csv(index=False).encode("utf-8"), file_name="transactions_rows.csv", mime="text/csv")

# ------------------ Optional: Add new row (writes to append sheet) ------------------
# Keep this optional write functionality but hide any sheet IDs / creds from UI.
if io_mod is not None:
    st.markdown("---")
    st.write("Add a new transaction (will write to the append sheet)")
    with st.expander("Add new row"):
        default_dt = start_sel if 'start_sel' in locals() else datetime.utcnow().date()
        new_date = st.date_input("Date", value=default_dt, min_value=min_date, max_value=max_date)
        bank_choice = st.selectbox("Bank", options=(banks_detected + ["Other (enter below)"]) if banks_detected else ["Other (enter below)"])
        bank_other = ""
        if bank_choice == "Other (enter below)":
            bank_other = st.text_input("Bank (custom)")
        txn_type = st.selectbox("Type", options=["debit", "credit"])
        amount = st.number_input("Amount (₹)", value=0.0, step=1.0)
        message = st.text_input("Message / Description", value="")
        submit_add = st.button("Save new row")
        if submit_add:
            chosen_bank = bank_other if bank_choice == "Other (enter below)" and bank_other else (bank_choice if bank_choice != "Other (enter below)" else "")
            now = datetime.utcnow()
            dt_combined = datetime.combine(new_date, now.time())
            new_row = {
                'DateTime': dt_combined.strftime("%Y-%m-%d %H:%M:%S"),
                'timestamp': dt_combined,
                'date': dt_combined.date(),
                'Bank': chosen_bank,
                'Type': txn_type,
                'Amount': amount,
                'Message': message,
                'is_deleted': 'false'
            }
            try:
                creds_info = _get_creds_info()
                res = io_mod.append_new_row(spreadsheet_id=SHEET_ID, range_name=APPEND_RANGE, new_row_dict=new_row, creds_info=creds_info, creds_file=None, history_range=RANGE)
                if res.get('status') == 'ok':
                    st.success("Appended new row to append sheet.")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to append row: {res.get('message')}")
            except Exception as e:
                st.error(f"Error while appending new row: {e}")
