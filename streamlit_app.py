# streamlit_app.py - Google Sheets only (secrets-based), robust chart handling + totals + soft-delete
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Spend", layout="wide")
st.title("💳 Daily Spending (Google Sheets)")

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

# ------------------ Read sheets ------------------
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

history_df = history_df if history_df is not None else pd.DataFrame()
append_df = append_df if append_df is not None else pd.DataFrame()

if history_df.empty and append_df.empty:
    st.error("No rows found in configured Google Sheets. Verify sheet names and service account permissions.")
    st.stop()

sheet_full_df = pd.concat([history_df, append_df], ignore_index=True, sort=False)
if '_sheet_row_idx' not in sheet_full_df.columns:
    sheet_full_df['_sheet_row_idx'] = pd.NA

# handle soft-deleted rows
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

# ------------------ Transform + bank detection ------------------
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

# ------------------ Sidebar Filters ------------------
st.sidebar.header("Filters")
banks_detected = sorted([b for b in converted_df['Bank'].unique() if pd.notna(b)])
sel_banks = st.sidebar.multiselect("Banks", options=banks_detected, default=banks_detected)

if sel_banks:
    converted_df_filtered = converted_df[converted_df['Bank'].isin(sel_banks)].copy()
else:
    converted_df_filtered = converted_df.copy()

# ------------------ Aggregation ------------------
with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(converted_df_filtered)

# ensure month_map exists even if merged is empty
month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}

# ------------------ Year / Month filters ------------------
if not merged.empty:
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    years = sorted(merged['Date'].dt.year.unique().tolist())
    years_opts = ['All'] + [str(y) for y in years]
    sel_year = st.sidebar.selectbox("Year", years_opts, index=0)

    if sel_year == 'All':
        month_frame = merged.copy()
    else:
        month_frame = merged[merged['Date'].dt.year == int(sel_year)]
    month_nums = sorted(month_frame['Date'].dt.month.unique().tolist())
    month_choices = [month_map[m] for m in month_nums]
    sel_months = st.sidebar.multiselect("Month(s)", options=month_choices, default=month_choices)
else:
    sel_year = 'All'
    sel_months = []

# ------------------ Determine date range ------------------
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

# ------------------ Date selection ------------------
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

# ------------------ Totals for selected date / range ------------------
try:
    start_sel, end_sel = selected_date_range_for_totals
    if isinstance(start_sel, datetime):
        start_sel = start_sel.date()
    if isinstance(end_sel, datetime):
        end_sel = end_sel.date()

    tmp_rows = converted_df_filtered.copy()
    if 'timestamp' in tmp_rows.columns:
        tmp_rows['timestamp'] = pd.to_datetime(tmp_rows['timestamp'], errors='coerce')
    elif 'date' in tmp_rows.columns:
        tmp_rows['timestamp'] = pd.to_datetime(tmp_rows['date'], errors='coerce')
    else:
        tmp_rows['timestamp'] = pd.NaT

    mask_sel = tmp_rows['timestamp'].dt.date.between(start_sel, end_sel)
    sel_df = tmp_rows.loc[mask_sel].copy()

    col_map_lower = {c.lower(): c for c in sel_df.columns}
    amount_col = col_map_lower.get('amount')
    type_col = col_map_lower.get('type')

    credit_sum = 0.0
    debit_sum = 0.0
    credit_count = 0
    debit_count = 0

    if not sel_df.empty and amount_col is not None:
        if type_col is not None:
            sel_df['type_norm'] = sel_df[type_col].astype(str).str.lower().str.strip()
            credit_mask = sel_df['type_norm'] == 'credit'
            debit_mask = sel_df['type_norm'] == 'debit'
            credit_sum = pd.to_numeric(sel_df.loc[credit_mask, amount_col], errors='coerce').fillna(0.0).sum()
            debit_sum = pd.to_numeric(sel_df.loc[debit_mask, amount_col], errors='coerce').fillna(0.0).sum()
            credit_count = int(credit_mask.sum())
            debit_count = int(debit_mask.sum())
        else:
            text_cols = [c for c in sel_df.columns if sel_df[c].dtype == object]
            for _, r in sel_df.iterrows():
                amt = pd.to_numeric(r.get(amount_col, 0), errors='coerce')
                if pd.isna(amt):
                    amt = 0.0
                txt = " ".join(str(r[c]) for c in text_cols if pd.notna(r[c])).lower()
                if 'credit' in txt:
                    credit_sum += amt
                    credit_count += 1
                else:
                    debit_sum += amt
                    debit_count += 1

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Credits ({start_sel} → {end_sel})", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    col2.metric(f"Debits ({start_sel} → {end_sel})", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    col3.metric("Net (Credits − Debits)", f"₹{(credit_sum - debit_sum):,.0f}")

except Exception as e:
    st.error(f"Failed to compute totals for selected date/range: {e}")

# ------------------ Apply year/month and date filters to aggregated plot_df ------------------
plot_df = merged.copy()
if sel_year != 'All':
    plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]
if sel_months:
    inv_map = {v: k for k, v in month_map.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ------------------ Chart & rendering (robust + fallback) ------------------
st.subheader("Daily Spend and Credit")
if plot_df.empty:
    st.info("No aggregated data available for selected filters.")
else:
    # Ensure Date is proper datetime and drop missing Dates
    try:
        plot_df['Date'] = pd.to_datetime(plot_df['Date'], errors='coerce')
    except Exception:
        pass
    plot_df = plot_df.dropna(subset=['Date']).copy()

    # Choose series
    show_debit = st.sidebar.checkbox("Show Debit (Total_Spent)", value=True)
    show_credit = st.sidebar.checkbox("Show Credit (Total_Credit)", value=True)
    series_selected = []
    if show_debit:
        series_selected.append('Total_Spent')
    if show_credit:
        series_selected.append('Total_Credit')

    existing_series = [c for c in series_selected if c in plot_df.columns]
    if not existing_series:
        numeric_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
        fallback = [c for c in ['Total_Spent', 'Total_Credit'] if c in numeric_cols]
        existing_series = fallback if fallback else numeric_cols[:2]

    if not existing_series:
        st.info("No numeric series available to plot.")
    else:
        # coerce to numeric, drop rows where all series are NaN
        for c in existing_series:
            plot_df[c] = pd.to_numeric(plot_df.get(c, 0), errors='coerce')
        plot_df = plot_df.dropna(subset=existing_series, how='all')
        # build safe_df indexed by Date
        try:
            safe_df = plot_df.set_index('Date')[existing_series].sort_index()
        except Exception:
            plot_df['Date_str'] = plot_df['Date'].dt.strftime('%Y-%m-%d')
            try:
                safe_df = plot_df.set_index('Date_str')[existing_series].sort_index()
            except Exception:
                safe_df = None

        # Try custom renderer; fallback to built-in line chart if it fails
        if charts_mod is not None:
            try:
                # pass copies to avoid side-effects
                charts_mod.render_chart(
                    plot_df=plot_df.reset_index(drop=True).copy(),
                    converted_df=converted_df_filtered.copy(),
                    chart_type="Daily line",
                    series_selected=existing_series,
                    top_n=5
                )
            except Exception as e:
                st.warning("Custom chart renderer failed — falling back to the built-in chart. " f"Error: {e}")
                if safe_df is not None and not safe_df.empty:
                    st.line_chart(safe_df)
                else:
                    try:
                        st.line_chart(plot_df.set_index('Date')[existing_series])
                    except Exception as e2:
                        st.error(f"Failed to plot fallback chart: {e2}")
        else:
            if safe_df is not None and not safe_df.empty:
                st.line_chart(safe_df)
            else:
                try:
                    st.line_chart(plot_df.set_index('Date')[existing_series])
                except Exception as e:
                    st.error(f"Failed to plot chart: {e}")

# ------------------ Rows view & download ------------------
st.subheader("Rows (matching selection)")
rows_df = converted_df_filtered.copy()
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
elif 'date' in rows_df.columns:
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
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")

    # Build selectable mapping
    selectable = False
    selectable_labels = []
    selectable_label_to_target = {}

    map_df = converted_df_filtered.copy()
    if '_sheet_row_idx' in map_df.columns and '_source_sheet' in map_df.columns:
        try:
            map_df['timestamp'] = pd.to_datetime(map_df['timestamp'], errors='coerce')
        except Exception:
            pass
        if start_sel and end_sel:
            map_df = map_df[(map_df['timestamp'].dt.date >= start_sel) & (map_df['timestamp'].dt.date <= end_sel)]
        for i, r in map_df.iterrows():
            ts = ''
            if 'timestamp' in r and pd.notna(r['timestamp']):
                try:
                    ts = pd.to_datetime(r['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    ts = str(r['timestamp'])
            amt = r.get('Amount', r.get('amount', ''))
            msg = r.get('Message', '') if 'Message' in r else (r.get('message', '') if 'message' in r else '')
            src = r.get('_source_sheet', 'history')
            idx = r.get('_sheet_row_idx')
            label = f"{i+1} | {src} | {ts} | {amt} | {str(msg)[:60]}"
            tgt_range = APPEND_RANGE if src == 'append' else RANGE
            try:
                if pd.isna(idx):
                    continue
            except Exception:
                pass
            selectable_labels.append(label)
            selectable_label_to_target[label] = (tgt_range, int(idx))
        if selectable_labels:
            selectable = True

    # Soft-delete UI
    if io_mod is not None and selectable:
        st.markdown("---")
        st.write("Bulk actions (Google Sheet only)")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            selected_labels = st.multiselect("Select rows to soft-delete", options=selectable_labels)
        with col_b:
            remove_btn = st.button("Remove selected rows", key="remove_rows_btn")

        if remove_btn:
            if not selected_labels:
                st.warning("No rows selected.")
            else:
                groups = {}
                for lbl in selected_labels:
                    tgt = selectable_label_to_target.get(lbl)
                    if not tgt:
                        continue
                    rng, idx = tgt
                    groups.setdefault(rng, []).append(idx)

                creds_info = _get_creds_info()
                any_error = False
                total_updated = 0
                for rng, indices in groups.items():
                    try:
                        res = io_mod.mark_rows_deleted(
                            spreadsheet_id=SHEET_ID,
                            range_name=rng,
                            creds_info=creds_info,
                            creds_file=None,
                            row_indices=indices
                        )
                        if isinstance(res, dict) and res.get('status') == 'ok':
                            total_updated += int(res.get('updated', 0))
                        else:
                            st.error(f"Failed to mark rows deleted in {rng}: {res}")
                            any_error = True
                    except Exception as e:
                        st.error(f"Error while marking rows deleted in {rng}: {e}")
                        any_error = True

                if not any_error:
                    st.success(f"Marked {total_updated} rows as deleted.")
                    st.experimental_rerun()

# ------------------ Optional: Add new row (writes to append sheet) ------------------
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
                res = io_mod.append_new_row(
                    spreadsheet_id=SHEET_ID,
                    range_name=APPEND_RANGE,
                    new_row_dict=new_row,
                    creds_info=creds_info,
                    creds_file=None,
                    history_range=RANGE
                )
                if isinstance(res, dict) and res.get('status') == 'ok':
                    st.success("Appended new row to append sheet.")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to append row: {res}")
            except Exception as e:
                st.error(f"Error while appending new row: {e}")
