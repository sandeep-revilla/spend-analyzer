# streamlit_app.py — Complete Daily Spend Tracker (Google Sheets + Charts + Add Row + Remove Row)
import streamlit as st
import pandas as pd
import importlib
import re
from datetime import datetime, timedelta, date, time as dt_time
import traceback
import os
from typing import Optional

st.set_page_config(page_title="💳 Daily Spend Tracker", layout="wide")
st.title("💳 Daily Spending")

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
        st.info("For local testing create .streamlit/secrets.toml with keys:\n\n"
                'SHEET_ID = "your-sheet-id"\nRANGE = "History Transactions"\nAPPEND_RANGE = "Append Transactions"\n\n'
                "Or set the keys in Streamlit Cloud app settings.")
        return

    # ------------------ Session State ------------------
    if "reload_key" not in st.session_state:
        st.session_state.reload_key = 0

    if "last_refreshed" not in st.session_state:
        st.session_state.last_refreshed = None

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
        """
        Reads the sheet and ensures two helper columns are present:
        - _sheet_row_idx: 0-based index of the data row (first data row after header is 0)
        - _source_sheet: 'history' or 'append' indicating which sheet the row came from
        """
        try:
            # Note: io_helpers.read_google_sheet should already use a wide A1 range fallback (A:AZ)
            df = io_mod.read_google_sheet(spreadsheet_id, range_name, creds_info=creds_info)
        except Exception as e:
            st.error(f"Failed to read sheet '{range_name}': {e}")
            st.text(traceback.format_exc())
            return pd.DataFrame()
        df = df.reset_index(drop=True)
        if not df.empty:
            # store the sheet-relative row index (first data row after header => 0)
            df["_sheet_row_idx"] = df.index.astype(int)
        df["_source_sheet"] = source_name
        return df

    # Reduced cache TTL to allow quick refreshes while testing appends.
    @st.cache_data(ttl=3, show_spinner=False)
    def fetch_sheets(spreadsheet_id, range_hist, range_append, creds_info, reload_key):
        hist_df = _read_sheet_with_index(spreadsheet_id, range_hist, "history", creds_info)
        app_df = _read_sheet_with_index(spreadsheet_id, range_append, "append", creds_info)
        return hist_df, app_df

    # ------------------ Refresh UI ------------------
    col_refresh, col_time = st.columns([1, 3])
    if col_refresh.button("🔁 Refresh Data", use_container_width=True):
        st.session_state.reload_key += 1
        st.experimental_rerun()

    if st.session_state.last_refreshed:
        col_time.caption(f"Last refreshed: {st.session_state.last_refreshed.strftime('%Y-%m-%d %H:%M:%S UTC')}")

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

    # keep df_raw (concatenation of both sheets) for possible mapping restoration
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

    # If transform dropped the internal mapping columns but the row counts match, restore them
    try:
        if ("_sheet_row_idx" not in converted_df.columns or "_source_sheet" not in converted_df.columns):
            if ("_sheet_row_idx" in df_raw.columns) and ("_source_sheet" in df_raw.columns) and (len(df_raw) == len(converted_df)):
                converted_df["_sheet_row_idx"] = df_raw["_sheet_row_idx"].reset_index(drop=True)
                converted_df["_source_sheet"] = df_raw["_source_sheet"].reset_index(drop=True)
    except Exception:
        # non-fatal: proceed without mapping if we can't restore
        pass

    # Ensure timestamp column is parsed early so we can compute month options
    if "timestamp" in converted_df.columns:
        converted_df["timestamp"] = pd.to_datetime(converted_df["timestamp"], errors="coerce")
    else:
        converted_df["timestamp"] = pd.to_datetime(converted_df.get("date"), errors="coerce")

    if "Bank" not in converted_df.columns:
        converted_df["Bank"] = "Unknown"

    # ------------------ Sidebar Filters ------------------
    st.sidebar.header("Filters")

    # Bank filter moved to sidebar
    banks = sorted(converted_df["Bank"].dropna().unique().tolist())
    sel_banks = st.sidebar.multiselect("Banks", options=banks, default=banks, key="bank_filter_sidebar")

    # Month filter (in sidebar)
    ts_valid = converted_df["timestamp"].dropna()
    if not ts_valid.empty:
        months = sorted(ts_valid.dt.to_period("M").astype(str).unique(), reverse=True)
    else:
        months = []
    month_options = ["All"] + months
    sel_month = st.sidebar.selectbox("Month", options=month_options, index=0,
                                     help="Select a calendar month to restrict the charts and totals to that month (or choose All).")

    # ------------------ Apply Filters ------------------
    filtered_df = converted_df[converted_df["Bank"].isin(sel_banks)].copy()

    if sel_month != "All" and sel_month:
        try:
            y, m = sel_month.split("-")
            y = int(y); m = int(m)
            mask_month = (filtered_df["timestamp"].dt.year == y) & (filtered_df["timestamp"].dt.month == m)
            filtered_df = filtered_df.loc[mask_month].copy()
        except Exception:
            pass

    # ------------------ Compute totals and charts ------------------
    with st.spinner("Computing daily totals..."):
        try:
            merged = transform.compute_daily_totals(filtered_df)
        except Exception as e:
            st.error("Error in compute_daily_totals(). See traceback below.")
            st.exception(e)
            st.text(traceback.format_exc())
            merged = pd.DataFrame()

    if not merged.empty:
        merged["Date"] = pd.to_datetime(merged["Date"]).dt.normalize()

    # ------------------ Date Range ------------------
    if "timestamp" not in filtered_df.columns:
        filtered_df["timestamp"] = pd.to_datetime(filtered_df.get("date"), errors="coerce")
    else:
        filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"], errors="coerce")

    valid_dates = filtered_df["timestamp"].dropna()
    if valid_dates.empty:
        st.warning("No valid dates found.")
        return

    min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    today = datetime.utcnow().date()

    if min_date > max_date:
        min_date, max_date = max_date, min_date

    default_sel_date = today
    if default_sel_date < min_date:
        default_sel_date = min_date
    if default_sel_date > max_date:
        default_sel_date = max_date

    totals_mode = st.sidebar.radio("Totals mode", ["Single date", "Date range"], index=0)

    if totals_mode == "Single date":
        sel_date = st.sidebar.date_input("Pick date", value=default_sel_date, min_value=min_date, max_value=max_date)
        start_sel, end_sel = sel_date, sel_date
    else:
        start_def, end_def = min_date, max_date
        if start_def < min_date:
            start_def = min_date
        if end_def > max_date:
            end_def = max_date
        if start_def > end_def:
            start_def = end_def
        dr = st.sidebar.date_input("Pick date range", value=(start_def, end_def), min_value=min_date, max_value=max_date)
        if isinstance(dr, (tuple, list)) and len(dr) == 2:
            start_sel, end_sel = dr
        else:
            start_sel, end_sel = dr, dr

    # ------------------ Totals ------------------
    tmp = filtered_df.copy()
    mask = tmp["timestamp"].dt.date.between(start_sel, end_sel)
    sel_df = tmp.loc[mask]

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

    c1, c2, c3 = st.columns(3)
    c1.metric("Credits", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    c2.metric("Debits", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    c3.metric("Net", f"₹{(credit_sum - debit_sum):,.0f}")

    # ------------------ Charts ------------------
    st.subheader("📊 Charts")
    if not merged.empty and charts_mod is not None:
        chart_type = st.selectbox("Chart type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=0)
        series_selected = []
        if st.sidebar.checkbox("Show Debit (Total_Spent)", True):
            series_selected.append("Total_Spent")
        if st.sidebar.checkbox("Show Credit (Total_Credit)", True):
            series_selected.append("Total_Credit")
        top_n = st.sidebar.slider("Top-N categories", 3, 20, 5)
        try:
            charts_mod.render_chart(merged, filtered_df, chart_type, series_selected, top_n=top_n, height=420)
        except Exception as e:
            st.error("Chart rendering failed. See traceback below.")
            st.exception(e)
            st.text(traceback.format_exc())
    else:
        st.info("No data for charts.")

    # ------------------ Rows Table ------------------
    st.subheader("Rows (matching selection)")
    rows_df = filtered_df.copy()
    mask = rows_df["timestamp"].dt.date.between(start_sel, end_sel)
    rows_df = rows_df.loc[mask]

    display_cols = [c for c in ["timestamp", "Bank", "Type", "Amount", "Message"] if c in rows_df.columns]
    st.dataframe(rows_df[display_cols], use_container_width=True, height=420)

    csv_data = rows_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_data, "transactions.csv", "text/csv")

    # ------------------ Remove Rows UI (soft delete) ------------------
    st.markdown("---")
    st.write("🗑️ Remove rows (soft delete)")

    if io_mod is None:
        st.info("Write operations (including remove) are disabled because io_helpers failed to import.")
    else:
        # Ensure we have mapping columns to the original sheets
        if "_sheet_row_idx" not in rows_df.columns or "_source_sheet" not in rows_df.columns:
            st.error("Missing internal row mapping columns ('_sheet_row_idx' or '_source_sheet'). Cannot delete rows.")
            st.info("These columns are normally added automatically when reading Google Sheets. Ensure io_helpers.read_google_sheet is used.")
        else:
            # Build selection choices labeled for readability
            def _label_for_row(r):
                ts = r.get("timestamp")
                bank = r.get("Bank", "")
                amt = r.get("Amount", "")
                msg = r.get("Message", "") or r.get("Message", "")
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
                "Select rows to remove (soft-delete). Selected rows will be marked is_deleted = TRUE.",
                options=labels_ordered,
                default=[]
            )

            selected_keys = []
            for lbl in selected_labels:
                for k in keys_ordered:
                    if choice_map[k]["label"] == lbl:
                        selected_keys.append(k)
                        break

            # Confirmation: require checkbox
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
                        st.experimental_rerun()

    # ------------------ Add New Row (date + time picker) ------------------
    st.markdown("---")
    st.write("➕ Add a new transaction")

    def build_timestamp_str(chosen_date: date, chosen_time: dt_time) -> str:
        """
        Format datetime in 'YYYY-MM-DD HH:MM:SS' using chosen date + chosen time.
        chosen_time is a datetime.time object; if None, use current UTC time.
        """
        if chosen_time is None:
            t = datetime.utcnow().time()
        else:
            t = chosen_time
        try:
            dt_combined = datetime.combine(chosen_date, t)
        except Exception:
            dt_combined = datetime.utcnow()
        return dt_combined.strftime("%Y-%m-%d %H:%M:%S"), dt_combined

    if io_mod is not None:
        with st.expander("Add new row"):
            with st.form("add_row_form", clear_on_submit=True):
                new_date = st.date_input("Date (picker)", value=datetime.utcnow().date())
                # new: time picker so user can specify the exact time that will be saved
                new_time = st.time_input("Time (UTC)", value=datetime.utcnow().time())
                bank = st.text_input("Bank", "HDFC Bank")
                txn_type = st.selectbox("Type", ["debit", "credit"])
                amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0, format="%.2f")
                msg = st.text_input("Message / Description", "")
                submit_add = st.form_submit_button("Save new row")

                if submit_add:
                    try:
                        timestamp_str, timestamp_dt = build_timestamp_str(new_date, new_time)

                        new_row = {
                            "DateTime": timestamp_str,
                            "timestamp": timestamp_dt,
                            "date": new_date.isoformat(),
                            "Bank": bank,
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
                            st.success("✅ Row added successfully with correct date & time!")
                            # force a quick refresh so the appended row is read back
                            st.session_state.reload_key += 1
                            st.experimental_rerun()
                        else:
                            st.error(f"Failed to add row: {res}")
                    except Exception as e:
                        st.error("Error adding row; see traceback below.")
                        st.exception(e)
                        st.text(traceback.format_exc())
    else:
        st.info("Write operations disabled: io_helpers module unavailable.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Unexpected error in app. See traceback below.")
        st.exception(e)
        st.text(traceback.format_exc())
