# streamlit_app.py — Complete Google Sheets integration + Totals + Charts + Add Row + Soft Delete + Refresh
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta
import altair as alt

st.set_page_config(page_title="Daily Spend Tracker", layout="wide")
st.title("💳 Daily Spending")

# ------------------ Module Imports ------------------
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import.")
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

# ------------------ Load Secrets ------------------
if not hasattr(st, "secrets") or not st.secrets:
    st.error("Streamlit secrets are not configured. Add SHEET_ID, RANGE, APPEND_RANGE, and gcp_service_account.")
    st.stop()

SHEET_ID = st.secrets.get("SHEET_ID")
RANGE = st.secrets.get("RANGE")
APPEND_RANGE = st.secrets.get("APPEND_RANGE")

if not SHEET_ID or not RANGE or not APPEND_RANGE:
    st.error("Missing required secrets: SHEET_ID, RANGE, or APPEND_RANGE.")
    st.stop()

# ------------------ Session State ------------------
if "reload_key" not in st.session_state:
    st.session_state.reload_key = 0
if "last_refreshed" not in st.session_state:
    st.session_state.last_refreshed = None

# ------------------ Refresh UI ------------------
col_refresh, col_time = st.columns([1, 3])
if col_refresh.button("🔁 Refresh Data", use_container_width=True):
    st.session_state.reload_key += 1
    st.experimental_rerun()

if st.session_state.last_refreshed:
    col_time.caption(f"Last refreshed at: {st.session_state.last_refreshed.strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ------------------ Google Auth Helpers ------------------
def _get_creds_info():
    if "gcp_service_account" in st.secrets and io_mod is not None:
        return io_mod.parse_service_account_secret(st.secrets["gcp_service_account"])
    return None

def _read_sheet_with_index(spreadsheet_id, range_name, source_name, creds_info):
    df = io_mod.read_google_sheet(spreadsheet_id, range_name, creds_info=creds_info, creds_file=None)
    df = df.reset_index(drop=True)
    if not df.empty:
        df['_sheet_row_idx'] = df.index.astype(int)  # 0-based relative to sheet data rows
    df['_source_sheet'] = source_name
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_sheets(spreadsheet_id, range_hist, range_append, creds_info, reload_key):
    hist_df = _read_sheet_with_index(spreadsheet_id, range_hist, "history", creds_info)
    app_df = _read_sheet_with_index(spreadsheet_id, range_append, "append", creds_info)
    return hist_df, app_df

# ------------------ Fetch Data ------------------
creds_info = _get_creds_info()
if creds_info is None:
    st.error("Missing Google service account credentials.")
    st.stop()

with st.spinner("Fetching Google Sheets..."):
    history_df, append_df = fetch_sheets(SHEET_ID, RANGE, APPEND_RANGE, creds_info, st.session_state.reload_key)
    st.session_state.last_refreshed = datetime.utcnow()

if history_df.empty and append_df.empty:
    st.error("No data found in the Google Sheets.")
    st.stop()

# Combine sheets
sheet_full_df = pd.concat([history_df, append_df], ignore_index=True, sort=False)

# ---------------- Safe UID creation helper ----------------
def _safe_col_as_str_series(df: pd.DataFrame, colname: str) -> pd.Series:
    """
    Return a pd.Series of strings for df[colname] if present; otherwise
    return a Series of empty strings of the same length as df.
    This avoids errors when using .astype on a fallback scalar.
    """
    if colname in df:
        # Convert values to string safely
        return df[colname].astype(object).fillna("").astype(str)
    # Return an empty-string series with the same length
    return pd.Series([""] * len(df), index=df.index, dtype=str)

# ---------- Soft-delete normalization + dedupe logic ----------
# Normalize is_deleted to boolean mask
if 'is_deleted' in sheet_full_df.columns:
    sheet_full_df['is_deleted_bool'] = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true','t','1','yes'])
else:
    sheet_full_df['is_deleted_bool'] = False

# Build deterministic _uid for a logical record so duplicates across sheets can be detected.
# Use timestamp/date + Bank + Type + Amount + Message (adjust field names if your sheet differs)
part_ts = _safe_col_as_str_series(sheet_full_df, 'timestamp')
part_date = _safe_col_as_str_series(sheet_full_df, 'date')
# prefer timestamp over date when both exist
time_series = part_ts.where(part_ts != "", part_date)
uid = (
    time_series.fillna("").astype(str).str.strip() + '|' +
    _safe_col_as_str_series(sheet_full_df, 'Bank').str.strip() + '|' +
    _safe_col_as_str_series(sheet_full_df, 'Type').str.strip() + '|' +
    _safe_col_as_str_series(sheet_full_df, 'Amount').str.strip() + '|' +
    _safe_col_as_str_series(sheet_full_df, 'Message').str.strip()
)
sheet_full_df['_uid'] = uid

# If ANY copy of a uid is marked deleted, remove all copies of that uid (prevents resurrection)
deleted_uids = sheet_full_df.loc[sheet_full_df['is_deleted_bool'], '_uid'].unique()
if len(deleted_uids) > 0:
    sheet_full_df = sheet_full_df.loc[~sheet_full_df['_uid'].isin(deleted_uids)].copy()

# Deduplicate duplicates (prefer history over append)
src_rank_map = {'history': 0, 'append': 1}
sheet_full_df['_src_rank'] = sheet_full_df['_source_sheet'].map(src_rank_map).fillna(1)
sheet_full_df = sheet_full_df.sort_values(['_uid', '_src_rank']).drop_duplicates('_uid', keep='first').reset_index(drop=True)

# Build the visible dataframe (df_raw) — after deletion/dedupe
df_raw = sheet_full_df.copy()

if df_raw.empty:
    st.warning("No visible data after filtering deleted rows.")
    st.stop()

# ------------------ Transform ------------------
converted_df = transform.convert_columns_and_derives(df_raw)

# ------------------ Bank Detection ------------------
def add_bank_column(df: pd.DataFrame) -> pd.DataFrame:
    bank_map = {"hdfc": "HDFC Bank", "indian bank": "Indian Bank"}
    if "Bank" in df.columns:
        df["Bank"] = df["Bank"].fillna("Unknown")
        return df
    df["Bank"] = df.astype(str).agg(" ".join, axis=1).str.lower().apply(
        lambda x: next((v for k, v in bank_map.items() if k in x), "Unknown")
    )
    return df

converted_df = add_bank_column(converted_df)

# ------------------ Sidebar Filters ------------------
st.sidebar.header("Filters")
banks = sorted(converted_df["Bank"].dropna().unique().tolist())
sel_banks = st.sidebar.multiselect("Banks", options=banks, default=banks)

filtered_df = converted_df[converted_df["Bank"].isin(sel_banks)] if sel_banks else converted_df

with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(filtered_df)

month_map = {i: pd.Timestamp(1900, i, 1).strftime("%B") for i in range(1, 13)}

if not merged.empty:
    merged["Date"] = pd.to_datetime(merged["Date"]).dt.normalize()
    years = sorted(merged["Date"].dt.year.unique().tolist())
    sel_year = st.sidebar.selectbox("Year", ["All"] + [str(y) for y in years], index=0)
    month_frame = merged if sel_year == "All" else merged[merged["Date"].dt.year == int(sel_year)]
    month_nums = sorted(month_frame["Date"].dt.month.unique().tolist())
    month_choices = [month_map[m] for m in month_nums]
    sel_months = st.sidebar.multiselect("Month(s)", options=month_choices, default=month_choices)
else:
    sel_year, sel_months = "All", []

# ------------------ Date Range ------------------ 
tmp = filtered_df.copy()
tmp["timestamp"] = pd.to_datetime(tmp.get("timestamp", tmp.get("date")), errors="coerce")
valid_dates = tmp["timestamp"].dropna()
if valid_dates.empty:
    st.error("No valid dates found.")
    st.stop()
min_date, max_date = valid_dates.min().date(), valid_dates.max().date()

totals_mode = st.sidebar.radio("Totals mode", ["Single date", "Date range"], index=0)
today = datetime.utcnow().date()
default_date = max(min_date, min(today, max_date))

if totals_mode == "Single date":
    sel_date = st.sidebar.date_input("Pick date", value=default_date, min_value=min_date, max_value=max_date)
    start_sel, end_sel = sel_date, sel_date
else:
    dr = st.sidebar.date_input("Pick start & end", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    start_sel, end_sel = dr if isinstance(dr, (tuple, list)) else (dr, dr)

# ------------------ Totals ------------------
try:
    tmp_rows = filtered_df.copy()
    tmp_rows["timestamp"] = pd.to_datetime(tmp_rows.get("timestamp", tmp_rows.get("date")), errors="coerce")
    mask = tmp_rows["timestamp"].dt.date.between(start_sel, end_sel)
    sel_df = tmp_rows.loc[mask].copy()

    amt_col = next((c for c in sel_df.columns if c.lower() == "amount"), None)
    type_col = next((c for c in sel_df.columns if c.lower() == "type"), None)
    credit_sum = debit_sum = credit_count = debit_count = 0
    if amt_col:
        sel_df["amt"] = pd.to_numeric(sel_df[amt_col], errors="coerce").fillna(0.0)
        if type_col:
            sel_df["type_norm"] = sel_df[type_col].astype(str).str.lower()
            credit_mask = sel_df["type_norm"] == "credit"
            debit_mask = sel_df["type_norm"] == "debit"
            credit_sum = sel_df.loc[credit_mask, "amt"].sum()
            debit_sum = sel_df.loc[debit_mask, "amt"].sum()
            credit_count, debit_count = int(credit_mask.sum()), int(debit_mask.sum())

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Credits ({start_sel} → {end_sel})", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    c2.metric(f"Debits ({start_sel} → {end_sel})", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    c3.metric("Net (Credits − Debits)", f"₹{(credit_sum - debit_sum):,.0f}")
except Exception as e:
    st.error(f"Failed to compute totals: {e}")

# ------------------ Charts ------------------
st.subheader("📊 Daily Spend and Credit")
plot_df = merged.copy()
if sel_year != "All":
    plot_df = plot_df[plot_df["Date"].dt.year == int(sel_year)]
if sel_months:
    inv_map = {v: k for k, v in month_map.items()}
    selected_months = [inv_map[m] for m in sel_months if m in inv_map]
    plot_df = plot_df[plot_df["Date"].dt.month.isin(selected_months)]

if not plot_df.empty:
    # We want Debit (Total_Spent) = red, Credit (Total_Credit) = green
    cols = []
    show_debit = st.sidebar.checkbox("Show Debit (Total_Spent)", True)
    show_credit = st.sidebar.checkbox("Show Credit (Total_Credit)", True)
    if show_debit: cols.append("Total_Spent")
    if show_credit: cols.append("Total_Credit")

    for col in ["Total_Spent", "Total_Credit"]:
        if col not in plot_df.columns:
            plot_df[col] = 0.0
    long = plot_df[["Date", "Total_Spent", "Total_Credit"]].melt(id_vars=["Date"],
                                                                  value_vars=["Total_Spent", "Total_Credit"],
                                                                  var_name="Type",
                                                                  value_name="Amount")
    long["Type"] = long["Type"].map({"Total_Spent": "Debit", "Total_Credit": "Credit"})
    if cols:
        desired_map = {"Total_Spent": "Debit", "Total_Credit": "Credit"}
        selected_types = [desired_map[c] for c in cols]
        long = long[long["Type"].isin(selected_types)]

    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Amount:Q", title="Amount (₹)"),
        color=alt.Color("Type:N",
                        scale=alt.Scale(domain=["Debit", "Credit"], range=["red", "green"]),
                        legend=alt.Legend(title="Type")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Type:N"), alt.Tooltip("Amount:Q", format=",")]
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No data for selected filters.")

# ------------------ Rows Table ------------------
st.subheader("Rows (matching selection)")
rows_df = filtered_df.copy()
rows_df["timestamp"] = pd.to_datetime(rows_df.get("timestamp", rows_df.get("date")), errors="coerce")
mask = rows_df["timestamp"].dt.date.between(start_sel, end_sel)
rows_df = rows_df.loc[mask]

display_cols = [c for c in ["timestamp", "Bank", "Type", "Amount", "Message", "Suspicious"] if c in rows_df.columns]
st.dataframe(rows_df[display_cols], use_container_width=True, height=420)
csv_data = rows_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv_data, "transactions.csv", "text/csv")

# ------------------ Soft Delete ------------------
if io_mod is not None:
    st.markdown("---")
    st.write("🗑️ Bulk actions (Soft Delete)")
    selectable_labels = []
    label_to_target = {}
    # For the labels we keep the mapping to the original sheet & original _sheet_row_idx
    for i, r in rows_df.iterrows():
        label = f"{i+1} | {r.get('Bank','')} | {r.get('timestamp','')} | ₹{r.get('Amount','')} | {r.get('Message','')[:50]}"
        src = r.get("_source_sheet", "history")
        idx = r.get("_sheet_row_idx", 0)
        tgt_range = APPEND_RANGE if src == "append" else RANGE
        selectable_labels.append(label)
        label_to_target[label] = (tgt_range, int(idx))
    selected_labels = st.multiselect("Select rows to delete", selectable_labels)
    if st.button("Remove selected rows"):
        total_updated = 0
        for lbl in selected_labels:
            rng, idx = label_to_target[lbl]
            res = io_mod.mark_rows_deleted(SHEET_ID, rng, creds_info=creds_info, row_indices=[idx])
            if res.get("status") == "ok":
                total_updated += res.get("updated", 0)
        st.success(f"Marked {total_updated} rows as deleted.")
        st.session_state.reload_key += 1
        st.experimental_rerun()

# ------------------ Add New Row (fixed: provide multiple date/timestamp keys) ------------------
if io_mod is not None:
    st.markdown("---")
    st.write("➕ Add a new transaction")
    with st.expander("Add new row"):
        new_date = st.date_input("Date", value=datetime.utcnow().date())
        bank = st.text_input("Bank", "HDFC Bank")
        txn_type = st.selectbox("Type", ["debit", "credit"])
        amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0)
        msg = st.text_input("Message / Description", "")
        submit_add = st.button("Save new row")
        if submit_add:
            dt_combined = datetime.combine(new_date, datetime.utcnow().time())
            # Provide multiple common column name variants for date/timestamp so the
            # append helper will find a matching header (DateTime, timestamp, date, Date)
            timestamp_str = dt_combined.strftime("%Y-%m-%d %H:%M:%S")
            date_str = dt_combined.date().isoformat()

            new_row = {
                # common timestamp column variants (sheet may use DateTime or timestamp)
                "DateTime": timestamp_str,
                "timestamp": timestamp_str,
                # common date-only column variants (sheet may use date or Date)
                "date": date_str,
                "Date": date_str,

                # other fields
                "Bank": bank,
                "Type": txn_type,
                "Amount": amount,
                "Message": msg,
                # ensure explicit is_deleted flag
                "is_deleted": "false",
            }

            res = io_mod.append_new_row(SHEET_ID, APPEND_RANGE, new_row, creds_info=creds_info, history_range=RANGE)
            if res.get("status") == "ok":
                st.success("✅ Row added successfully!")
                st.session_state.reload_key += 1
                st.experimental_rerun()
            else:
                st.error(f"Failed to add row: {res}")
