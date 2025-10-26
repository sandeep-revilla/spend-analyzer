# streamlit_app.py — Complete Daily Spend Tracker (Google Sheets + Charts + Add Row + Remove Row)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, date, time as dt_time, timedelta
import traceback
from typing import Optional
import math

st.set_page_config(page_title="💳 Daily Spend Tracker", layout="wide")
st.title("💳 Daily Spending")


# --- NEW: Helper Function to Calculate Running Balance (from 0) ---
def calculate_running_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a running balance in-memory, assuming a starting balance of 0
    for each bank before its first transaction.
    """
    df = df.copy()
    
    # Ensure a valid timestamp column exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT # No date, can't balance

    df = df.dropna(subset=['timestamp'])
    if df.empty:
        df['Balance'] = pd.NA
        return df

    # Ensure essential columns are numeric/clean
    df['Amount'] = pd.to_numeric(df.get('Amount'), errors='coerce').fillna(0.0)
    df['Type_norm'] = df.get('Type', 'debit').astype(str).str.lower()
    df['Bank'] = df.get('Bank', 'Unknown').astype(str).str.strip()

    # Create a 'signed' amount for credits (+) and debits (-)
    df['Signed_Amount'] = df.apply(
        lambda r: r['Amount'] if r['Type_norm'] == 'credit' else -r['Amount'],
        axis=1
    )
    
    # Sort by bank, then time, to ensure the cumsum is in the correct order
    df = df.sort_values(by=['Bank', 'timestamp'])
    
    # The running balance is the cumulative sum, grouped by bank
    df['Balance'] = df.groupby('Bank')['Signed_Amount'].cumsum()
    
    return df
# --- END NEW FUNCTION ---


def main():
    # ------------------ Module Imports ------------------
    try:
        transform = importlib.import_module("transform")
    except Exception as e:
        st.error("❌ Missing or broken transform.py.")
        st.exception(e)
        st.text(traceback.format_exc())
        return

    try:
        import io_helpers as io_mod
    except Exception as e:
        st.warning("⚠️ io_helpers import failed — write operations disabled. Continuing in read-only mode.")
        st.exception(e)
        io_mod = None

    try:
        charts_mod = importlib.import_module("charts")
    except Exception:
        charts_mod = None

    # ------------------ Load from Streamlit Secrets ------------------
    _secrets = getattr(st, "secrets", {}) or {}
    SHEET_ID = _secrets.get("SHEET_ID")
    RANGE = _secrets.get("RANGE")
    APPEND_RANGE = _secrets.get("APPEND_RANGE")

    if not SHEET_ID or not RANGE or not APPEND_RANGE:
        st.error("Missing Google Sheet secrets: SHEET_ID, RANGE, APPEND_RANGE.")
        st.info(
            "For local testing create .streamlit/secrets.toml with keys:\n\n"
            'SHEET_ID = "your-sheet-id"\nRANGE = "History Transactions"\nAPPEND_RANGE = "Append Transactions"\n\n'
            "Or set the keys in Streamlit Cloud app settings."
        )
        return

    # ------------------ Session State ------------------
    if "reload_key" not in st.session_state:
        st.session_state.reload_key = 0
    if "last_refreshed" not in st.session_state:
        st.session_state.last_refreshed = None
    # FIX: 'all_bank_options' stores the list of available banks
    # 'selected_banks_filter' stores the user's selection
    if "all_bank_options" not in st.session_state:
        st.session_state.all_bank_options = []
    if "selected_banks_filter" not in st.session_state:
        st.session_state.selected_banks_filter = []


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

    def _read_sheet_with_index(spreadsheet_id: str, range_name: str, source_name: str, creds_info: Optional[dict]):
        try:
            df = io_mod.read_google_sheet(spreadsheet_id, range_name, creds_info=creds_info)
        except Exception as e:
            st.error(f"Failed to read sheet '{range_name}': {e}")
            st.text(traceback.format_exc())
            return pd.DataFrame()
        df = df.reset_index(drop=True)
        if not df.empty:
            df["_sheet_row_idx"] = df.index.astype(int)
        df["_source_sheet"] = source_name
        return df

    # --- FIX: Changed ttl from 3 to 300 seconds (5 minutes) ---
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_sheets(spreadsheet_id, range_hist, range_append, creds_info, reload_key):
        hist_df = _read_sheet_with_index(spreadsheet_id, range_hist, "history", creds_info)
        app_df = _read_sheet_with_index(spreadsheet_id, range_append, "append", creds_info)
        return hist_df, app_df

    # ------------------ Sidebar: Refresh + Controls ------------------
    st.sidebar.header("Controls")
    st.sidebar.markdown(
        """
        <style>
         div[data-testid="stSidebar"] button[data-testid="stButton"] {
            background-color: #2ecc71;
            color: white;
         }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🔁 Refresh Data", use_container_width=True, key="refresh_button"):
        st.session_state.reload_key += 1
        st.rerun() # --- FIX: Replaced st.experimental_rerun() ---

    if st.session_state.last_refreshed:
        st.caption(f"Last refreshed: {st.session_state.last_refreshed.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # ------------------ Fetch Data ------------------
    creds_info = _get_creds_info()
    if creds_info is None:
        st.error("Missing Google credentials (gcp_service_account).")
        st.info("If you don't need write access, add a dummy gcp_service_account or comment out write code while debugging.")
        return

    with st.spinner("Fetching Google Sheets..."):
        history_df, append_df = fetch_sheets(SHEET_ID, RANGE, APPEND_RANGE, creds_info, st.session_state.reload_key)
        st.session_state.last_refreshed = datetime.utcnow()

    if history_df.empty and append_df.empty:
        st.error("No data found in the Google Sheet.")
        return

    df_raw = pd.concat([history_df, append_df], ignore_index=True, sort=False)

    # ------------------ Filter Deleted ------------------
    if "is_deleted" in df_raw.columns:
        mask = df_raw["is_deleted"].astype(str).str.lower().isin(["true", "t", "1", "yes"])
        df_raw = df_raw.loc[~mask].copy()

    if df_raw.empty:
        st.warning("No visible rows (after filtering deleted entries).")
        return

    # ------------------ Transform ------------------
    try:
        converted_df = transform.convert_columns_and_derives(df_raw.copy())
    except Exception as e:
        st.error("Error while transforming sheet data (convert_columns_and_derives). See traceback below.")
        st.exception(e)
        st.text(traceback.format_exc())
        return

    # restore mapping cols if transform removed them
    try:
        if ("_sheet_row_idx" not in converted_df.columns or "_source_sheet" not in converted_df.columns):
            if ("_sheet_row_idx" in df_raw.columns) and ("_source_sheet" in df_raw.columns) and (len(df_raw) == len(converted_df)):
                converted_df["_sheet_row_idx"] = df_raw["_sheet_row_idx"].reset_index(drop=True)
                converted_df["_source_sheet"] = df_raw["_source_sheet"].reset_index(drop=True)
    except Exception:
        pass

    if "timestamp" in converted_df.columns:
        converted_df["timestamp"] = pd.to_datetime(converted_df["timestamp"], errors="coerce")
    else:
        converted_df["timestamp"] = pd.to_datetime(converted_df.get("date"), errors="coerce")

    if "Bank" not in converted_df.columns:
        converted_df["Bank"] = "Unknown"

    # --- NEW: Calculate running balance ---
    with st.spinner("Calculating running balances..."):
        converted_df_with_balance = calculate_running_balance(converted_df)
    # --- END NEW ---
    

    # --- NEW: Balance Display (shows the *latest* balance) ---
    st.subheader("🏦 Current Balances")

    latest_balances_df = pd.DataFrame()
    if not converted_df_with_balance.empty and 'timestamp' in converted_df_with_balance.columns and not converted_df_with_balance['timestamp'].isnull().all():
        latest_balances_df = converted_df_with_balance.loc[
            converted_df_with_balance.groupby('Bank')['timestamp'].idxmax(skipna=True)
        ]

    total_balance = 0.0
    if not latest_balances_df.empty and 'Balance' in latest_balances_df.columns:
        banks_to_show = sorted(latest_balances_df['Bank'].unique())
        if not banks_to_show:
             st.info("No bank data found to calculate balances.")
        else:
            balance_cols = st.columns(len(banks_to_show) + 1)
            
            for i, bank_name in enumerate(banks_to_show):
                row = latest_balances_df[latest_balances_df['Bank'] == bank_name].iloc[0]
                current_balance = row['Balance']
                
                if pd.notna(current_balance):
                    total_balance += current_balance
                    with balance_cols[i]:
                        st.metric(f"{bank_name} Balance", f"₹{current_balance:,.0f}")
                else:
                    with balance_cols[i]:
                        st.metric(f"{bank_name} Balance", "N/A")

            # Show Total Balance
            with balance_cols[-1]:
                st.metric("Total Balance", f"₹{total_balance:,.0f}", "All Accounts")
    else:
        st.info("Calculating balances... (or no data found)")

    st.markdown("---")
    # --- END NEW BALANCE SECTION ---


    # ------------------ Compute global daily totals ------------------
    try:
        merged_all = transform.compute_daily_totals(converted_df_with_balance.copy())
        if not merged_all.empty:
            merged_all["Date"] = pd.to_datetime(merged_all["Date"]).dt.normalize()
    except Exception:
        merged_all = pd.DataFrame()

    # ------------------ Sidebar Filters (RE-GROUPED) ------------------
    st.sidebar.header("Filters")
    
    # Get all available banks for filter options
    banks_available = sorted([b for b in converted_df_with_balance["Bank"].dropna().unique().tolist()])
    if not banks_available:
        banks_available = ["Unknown"]
    
    # Store this list in session state for the 'Add Row' form
    st.session_state.all_bank_options = banks_available.copy()

    # --- Expander 1: Chart & Metric Options ---
    with st.sidebar.expander("📊 Chart & Metric Options", expanded=False):
        ts_valid = converted_df_with_balance["timestamp"].dropna()
        years = sorted(ts_valid.dt.year.unique().tolist(), reverse=True) if not ts_valid.empty else []
        year_options = ["All"] + [str(y) for y in years]
        sel_year = st.selectbox("Year (for monthly average)", options=year_options, index=0)

        month_map = {i: pd.Timestamp(1900, i, 1).strftime("%B") for i in range(1, 13)}
        month_options = ["All"] + [month_map[i] for i in range(1, 13)]
        sel_month_name = st.selectbox("Month (for monthly average)", options=month_options, index=0)

        exclude_outliers = st.checkbox("Exclude outliers from average", value=False, key="exclude_outliers")
        show_debit = st.checkbox("Show Debit (Total_Spent)", value=True, key="show_debit")
        show_credit = st.checkbox("Show Credit (Total_Credit)", value=False, key="show_credit")
        chart_type = st.selectbox("Chart type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=0)
        top_n = st.slider("Top-N categories", 3, 20, 5, key="top_n")

    # --- Expander 2: Transaction Filters ---
    with st.sidebar.expander("🔍 Transaction Filters", expanded=True):
        
        # --- FIX: Set default selection only on the first run ---
        if "selected_banks_filter" not in st.session_state:
            st.session_state.selected_banks_filter = banks_available.copy()

        sel_banks = st.multiselect(
            "Banks", 
            options=banks_available, 
            key="selected_banks_filter" # This key links to the persistent selection
        )
        
        totals_mode = st.radio("Totals mode", ["Single date", "Date range"], index=0)

        # --- Date Range (for rows & totals) ---
        valid_dates_all = converted_df_with_balance["timestamp"].dropna()
        if valid_dates_all.empty:
            st.warning("No valid dates found in data.")
            min_date, max_date = date.today() - timedelta(days=30), date.today()
        else:
            min_date, max_date = valid_dates_all.min().date(), valid_dates_all.max().date()

        if min_date > max_date:
            min_date, max_date = max_date, min_date
        
        today = datetime.utcnow().date()
        default_sel_date = min(max(today, min_date), max_date) # Default to today, clamped within range

        if totals_mode == "Single date":
            sel_date = st.date_input("Pick date", value=default_sel_date, min_value=min_date, max_value=max_date)
            start_sel, end_sel = sel_date, sel_date
        else:
            default_start = min_date
            default_end = max_date
            dr = st.date_input("Pick date range", value=(default_start, default_end), min_value=min_date, max_value=max_date)
            if isinstance(dr, (tuple, list)) and len(dr) == 2:
                start_sel, end_sel = dr
            else:
                start_sel, end_sel = dr, dr # Fallback if only one date is picked
            
            if start_sel > end_sel:
                start_sel, end_sel = end_sel, start_sel
    

    # ------------------ Apply Filters ------------------
    # Use the 'Balance' inclusive dataframe
    filtered_df = converted_df_with_balance[converted_df_with_balance["Bank"].isin(sel_banks)].copy()

    # compute filtered daily totals for charts
    with st.spinner("Computing daily totals..."):
        try:
            merged = transform.compute_daily_totals(filtered_df.copy())
        except Exception as e:
            st.error("Error in compute_daily_totals(). See traceback below.")
            st.exception(e)
            st.text(traceback.format_exc())
            merged = pd.DataFrame()

    if not merged.empty:
        merged["Date"] = pd.to_datetime(merged["Date"]).dt.normalize()


    # ------------------ Compute totals for selected date/range ------------------
    tmp = filtered_df.copy()
    mask = tmp["timestamp"].dt.date.between(start_sel, end_sel)
    sel_df = tmp.loc[mask] # This is the final filtered DF for table + totals

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
            credit_count = int(credit_mask.sum())
            debit_count = int(debit_mask.sum())

    # ------------------ Monthly average metric logic ------------------
    def _safe_mean(s):
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        return float(s2.mean()) if not s2.empty else None

    def _compute_month_avg_from_merged(merged_df, year, month, replace_outliers=False):
        if merged_df is None or merged_df.empty:
            return None, 0, {"reason": "no_data"}
        df = merged_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        mask = (df["Date"].dt.year == int(year)) & (df["Date"].dt.month == int(month))
        dfm = df.loc[mask].copy()
        if dfm.empty:
            return None, 0, {"reason": "no_rows_for_month"}
        vals = pd.to_numeric(dfm.get("Total_Spent", 0), errors="coerce").fillna(0.0)
        if not replace_outliers or len(vals) < 3:
            return _safe_mean(vals), int(len(vals)), {"outliers_replaced": 0, "n": len(vals)}
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        is_out = (vals < lower) | (vals > upper)
        non_out = vals[~is_out]
        replacement = float(non_out.median()) if not non_out.empty else float(vals.median())
        vals_repl = vals.copy()
        vals_repl[is_out] = replacement
        return _safe_mean(vals_repl), int(len(vals_repl)), {"outliers_replaced": int(is_out.sum()), "n": len(vals_repl)}

    metric_year = None
    metric_month = None
    if sel_year != "All" and sel_month_name != "All":
        try:
            metric_year = int(sel_year)
            inv_map = {v: k for k, v in month_map.items()}
            metric_month = inv_map.get(sel_month_name)
        except Exception:
            metric_year = metric_month = None
    else:
        if not merged_all.empty:
            latest = merged_all["Date"].max()
            metric_year = int(latest.year)
            metric_month = int(latest.month)

    metric_avg = metric_count = prev_avg = prev_count = None
    if metric_year is not None and metric_month is not None:
        try:
            metric_avg, metric_count, _ = _compute_month_avg_from_merged(merged_all, metric_year, metric_month, replace_outliers=exclude_outliers)
            prev_dt = datetime(metric_year, metric_month, 1) - pd.DateOffset(months=1)
            prev_avg, prev_count, _ = _compute_month_avg_from_merged(merged_all, int(prev_dt.year), int(prev_dt.month), replace_outliers=exclude_outliers)
        except Exception:
            metric_avg = prev_avg = None

    # ------------------ Top-right compact metric ------------------
    top_left, top_mid, top_right = st.columns([6, 2, 2])
    with top_right:
        title_small = "AVG spent"
        metric_text = f"₹{metric_avg:,.2f}" if metric_avg is not None else "N/A"

        if prev_avg is None or metric_avg is None:
            delta_html = "<div style='text-align:right; color:gray; font-weight:600'>N/A</div>"
        else:
            diff = metric_avg - prev_avg
            try:
                delta_label = f"{(diff / abs(prev_avg) * 100.0):+.1f}%" if abs(prev_avg) > 1e-9 else f"{diff:+.2f}"
            except Exception:
                delta_label = f"{diff:+.2f}"
            
            color = "red" if diff > 0 else ("green" if diff < 0 else "gray")
            arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "►")
            delta_html = f"<div style='text-align:right; color:{color}; font-weight:700'>{arrow} {delta_label}</div>"

        month_label_display = pd.Timestamp(metric_year, metric_month, 1).strftime("%b-%y") if metric_year and metric_month else "—"

        st.markdown(
            "<div style='text-align:right'>"
            f"<div style='font-size:12px;color:#666'>{title_small}</div>"
            f"<div style='font-size:14px;color:#444'>{month_label_display}</div>"
            f"<div style='font-size:20px;font-weight:800'>{metric_text}</div>"
            f"{delta_html}"
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------ Charts ------------------
    st.subheader("📊 Charts")
    if not merged.empty and charts_mod is not None:
        series_selected = []
        if show_debit:
            series_selected.append("Total_Spent")
        if show_credit:
            series_selected.append("Total_Credit")
        try:
            charts_mod.render_chart(merged, filtered_df, chart_type, series_selected, top_n=top_n, height=420)
        except Exception as e:
            st.error("Chart rendering failed. See traceback below.")
            st.exception(e)
            st.text(traceback.format_exc())
    else:
        st.info("No data for charts (or charts.py is missing).")

    # ------------------ Rows Table (MODIFIED) ------------------
    st.subheader("Rows (matching selection)")
    rows_df = sel_df.copy() # Use the date-filtered dataframe

    _desired = ['timestamp', 'bank', 'type', 'amount', 'balance', 'message'] # Added 'balance'
    col_map = {c.lower(): c for c in rows_df.columns}
    display_cols = [col_map[d] for d in _desired if d in col_map]
    
    if not any(c.lower() == 'timestamp' for c in display_cols) and 'date' in col_map: 
        display_cols.insert(0, col_map['date'])

    if not display_cols: # Fallback
        st.warning("Could not find preferred columns - showing raw data."); 
        display_df = rows_df
    else:
        display_df = rows_df[display_cols].copy()
        
        # Formatting
        ts_col = next((c for c in display_df.columns if c.lower() in ['timestamp', 'date']), None)
        if ts_col: display_df[ts_col] = pd.to_datetime(display_df[ts_col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M').fillna('')
        
        amt_col = next((c for c in display_df.columns if c.lower() == 'amount'), None)
        if amt_col: display_df[amt_col] = pd.to_numeric(display_df[amt_col], errors='coerce')
        
        bal_col = next((c for c in display_df.columns if c.lower() == 'balance'), None)
        if bal_col: display_df[bal_col] = pd.to_numeric(display_df[bal_col], errors='coerce')

        pretty_rename = {
            'timestamp':'Timestamp','date':'Timestamp','bank':'Bank','type':'Type',
            'amount':'Amount','balance':'Balance','message':'Message'
        }
        display_df = display_df.rename(columns={c:pretty_rename[c.lower()] for c in display_df.columns if c.lower() in pretty_rename})
        
        final_order = [c for c in ['Timestamp', 'Bank', 'Type', 'Amount', 'Balance', 'Message'] if c in display_df.columns]
        display_df = display_df[final_order]

    # Sort by timestamp descending for display
    if 'Timestamp' in display_df.columns:
        display_df = display_df.sort_values(by='Timestamp', ascending=False)

    st.dataframe(
        display_df.reset_index(drop=True), 
        use_container_width=True, 
        height=420,
        column_config={
            "Amount": st.column_config.NumberColumn(format="₹%.2f"),
            "Balance": st.column_config.NumberColumn(format="₹%.0f") # Format balance
        }
    )
    # --- END MODIFIED TABLE ---

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_data, "transactions.csv", "text/csv")

    # ------------------ Remove Rows UI (soft delete) ------------------
    st.markdown("---")
    st.write("🗑️ Remove rows (soft delete)")

    if io_mod is None:
        st.info("Write operations (including remove) are disabled because io_helpers failed to import.")
    else:
        # Use rows_df, which is already filtered by bank + date
        if "_sheet_row_idx" not in rows_df.columns or "_source_sheet" not in rows_df.columns:
            st.error("Missing internal row mapping columns ('_sheet_row_idx' or '_source_sheet'). Cannot delete rows.")
            st.info("These columns are normally added automatically when reading Google Sheets. Ensure io_helpers.read_google_sheet is used.")
        else:
            def _label_for_row(r):
                ts = r.get("timestamp")
                bank = r.get("Bank", "")
                amt = r.get("Amount", "")
                msg = r.get("Message", "") or r.get("message", "")
                src = r.get("_source_sheet", "")
                idx = r.get("_sheet_row_idx", "")
                msg_short = (str(msg)[:40] + "...") if msg and len(str(msg)) > 40 else str(msg)
                return f"{src}:{idx} — {ts} | {bank} | {amt} | {msg_short}"

            choice_map = {}
            for _, row in rows_df.iterrows():
                key = f"{row['_source_sheet']}:{int(row['_sheet_row_idx'])}"
                label = _label_for_row(row)
                choice_map[key] = {"label": label, "source": row["_source_sheet"], "idx": int(row["_sheet_row_idx"])}

            keys_ordered = list(choice_map.keys())
            labels_ordered = [choice_map[k]["label"] for k in keys_ordered]

            selected_labels = st.multiselect(
                "Selected rows will be marked deleted",
                options=labels_ordered,
                default=[]
            )

            selected_keys = []
            for lbl in selected_labels:
                for k in keys_ordered:
                    if choice_map[k]["label"] == lbl:
                        selected_keys.append(k)
                        break

            confirm = st.checkbox("I confirm I want to mark the selected rows as deleted")

            if st.button("Mark selected rows deleted"):
                if not selected_keys:
                    st.warning("No rows selected to remove.")
                elif not confirm:
                    st.warning("Please confirm deletion by checking the box.")
                else:
                    grouped = {}
                    for k in selected_keys:
                        src, idx_s = k.split(":")
                        grouped.setdefault(src, []).append(int(idx_s))
                    overall_updated = 0
                    errors = []
                    for src, idxs in grouped.items():
                        tgt_range = RANGE if src == "history" else APPEND_RANGE
                        try:
                            res = io_mod.mark_rows_deleted(spreadsheet_id=SHEET_ID, range_name=tgt_range,
                                                           creds_info=creds_info, creds_file=None,
                                                           row_indices=idxs)
                            if res.get("status") == "ok":
                                overall_updated += int(res.get("updated", 0))
                            else:
                                errors.append(f"{src}: {res.get('message')}")
                        except Exception as e:
                            errors.append(f"{src}: {e}")

                    if errors:
                        st.error("Some deletions failed:")
                        for e in errors:
                            st.text(e)
                    else:
                        st.success(f"Marked {overall_updated} rows as deleted.")
                        st.session_state.reload_key += 1
                        st.rerun() # --- FIX: Replaced st.experimental_rerun() ---

    # ------------------ Add New Row ------------------
    st.markdown("---")
    st.write("➕ Add a new transaction")

    def build_timestamp_str_using_now(chosen_date: date):
        now = datetime.utcnow()
        try:
            dt_combined = datetime.combine(chosen_date, now.time())
        except Exception:
            dt_combined = now
        formatted = dt_combined.strftime("%m/%d/%Y %H:%M")
        return formatted, dt_combined

    if io_mod is not None:
        with st.expander("Add new row"):
            with st.form("add_row_form", clear_on_submit=True):
                new_date = st.date_input("Date (picker)", value=datetime.utcnow().date())

                # --- FIX: Use the 'all_bank_options' list from session state ---
                add_bank_options = st.session_state.all_bank_options.copy()
                add_bank_options = sorted(list(set(add_bank_options)))

                chosen_bank_sel = st.selectbox("Bank (from existing)", options=add_bank_options, index=0, key="add_bank_select")
                new_bank_input = st.text_input("New Bank (Optional - overrides dropdown)", value="", key="add_bank_new")

                txn_type = st.selectbox("Type", ["debit", "credit"])
                amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0, format="%.2f")
                msg = st.text_input("Message / Description", "")
                submit_add = st.form_submit_button("Save new row")

                if submit_add:
                    try:
                        new_bank_name = new_bank_input.strip()
                        chosen_bank = new_bank_name if new_bank_name else (chosen_bank_sel if chosen_bank_sel else "Unknown")
                        if not chosen_bank:
                            chosen_bank = "Unknown"

                        timestamp_str, timestamp_dt = build_timestamp_str_using_now(new_date)

                        new_row = {
                            "DateTime": timestamp_str,
                            "timestamp": timestamp_dt,
                            "date": timestamp_dt.strftime("%m/%d/%Y %H:%M"),
                            "Bank": chosen_bank,
                            "Type": txn_type,
                            "Amount": amount,
                            "Message": msg,
                            "is_deleted": "false",
                        }
                        
                        res = io_mod.append_new_row(
                            spreadsheet_id=SHEET_ID,
                            range_name=APPEND_RANGE,
                            new_row_dict=new_row,
                            creds_info=creds_info,
                            history_range=RANGE,
                        )

                        if res.get("status") == "ok":
                            st.success("✅ Row added successfully!")
                            st.session_state.reload_key += 1
                            # Clear selection cache to default to all (including new bank)
                            if "selected_banks_filter" in st.session_state:
                                del st.session_state.selected_banks_filter
                            st.rerun() # --- FIX: Replaced st.experimental_rerun() ---
                        else:
                            st.error(f"Failed to add row: {res}")
                    except Exception as e:
                        st.error("Error adding row; see traceback below.")
                        st.exception(e)
                        st.text(traceback.format_exc())
    else:
        st.info("Write operations disabled: io_helpers module unavailable.")

    # ------------------ Summary (Selected date/range) ------------------
    st.markdown("---")
    st.subheader("Summary (Selected date/range)")
    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    c1.metric("Credits", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    c2.metric("Debits", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    c3.metric("Net (Credits − Debits)", f"₹{(credit_sum - debit_sum):,.0f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Unexpected error in app. See traceback below.")
        st.exception(e)
        st.text(traceback.format_exc())
