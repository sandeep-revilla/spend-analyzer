# streamlit_app.py — Complete Daily Spend Tracker (Google Sheets + Charts + Add Row + Remove Row)
import streamlit as st
import pandas as pd
import importlib
import re
from datetime import datetime, timedelta, date, time as dt_time
import traceback
import os
from typing import Optional
import numpy as np

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

    # We'll keep an in-memory bank list so we can add new banks immediately after append
    if "bank_options" not in st.session_state:
        st.session_state["bank_options"] = None

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

    # Reduced cache TTL to allow quick refreshes while testing appends.
    @st.cache_data(ttl=3, show_spinner=False)
    def fetch_sheets(spreadsheet_id, range_hist, range_append, creds_info, reload_key):
        hist_df = _read_sheet_with_index(spreadsheet_id, range_hist, "history", creds_info)
        app_df = _read_sheet_with_index(spreadsheet_id, range_append, "append", creds_info)
        return hist_df, app_df

    # ------------------ Sidebar: Refresh (green) + Controls ------------------
    st.sidebar.header("Controls")

    # Green refresh button in sidebar (best-effort styling)
    st.sidebar.markdown(
        """
        <style>
         /* style the sidebar button (best-effort; may affect other buttons) */
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
        st.experimental_rerun()

    # show last refreshed text in main header area
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
        pass

    # Ensure timestamp column is parsed early so we can compute month / year options
    if "timestamp" in converted_df.columns:
        converted_df["timestamp"] = pd.to_datetime(converted_df["timestamp"], errors="coerce")
    else:
        converted_df["timestamp"] = pd.to_datetime(converted_df.get("date"), errors="coerce")

    if "Bank" not in converted_df.columns:
        converted_df["Bank"] = "Unknown"

    # ------------------ Compute global (unfiltered) daily totals for metric ------------------
    try:
        merged_all = transform.compute_daily_totals(converted_df.copy())
        if not merged_all.empty:
            merged_all["Date"] = pd.to_datetime(merged_all["Date"]).dt.normalize()
    except Exception:
        merged_all = pd.DataFrame()

    # ------------------ Sidebar Filters ------------------
    st.sidebar.header("Filters")

    # Bank filter computed dynamically from data
    banks = sorted([b for b in converted_df["Bank"].dropna().unique().tolist()])
    if not banks:
        banks = ["Unknown"]

    # initialize session_state bank_options if not set
    if st.session_state.get("bank_options") is None:
        st.session_state["bank_options"] = banks.copy()
    else:
        # ensure session state contains all banks present in data (in case history had banks not present earlier)
        for b in banks:
            if b not in st.session_state["bank_options"]:
                st.session_state["bank_options"].append(b)
        st.session_state["bank_options"] = sorted(list(set(st.session_state["bank_options"])))

    sel_banks = st.sidebar.multiselect("Banks", options=st.session_state["bank_options"], default=st.session_state["bank_options"], key="bank_filter_sidebar")

    # Year & Month filters for monthly average (separate controls)
    ts_valid = converted_df["timestamp"].dropna()
    years = sorted(ts_valid.dt.year.unique().tolist(), reverse=True) if not ts_valid.empty else []
    year_options = ["All"] + [str(y) for y in years]
    sel_year = st.sidebar.selectbox("Year (for monthly average)", options=year_options, index=0)

    month_map = {i: pd.Timestamp(1900, i, 1).strftime("%B") for i in range(1, 13)}
    month_options = ["All"] + [month_map[i] for i in range(1, 13)]
    sel_month_name = st.sidebar.selectbox("Month (for monthly average)", options=month_options, index=0)

    # Exclude outliers checkbox (simple text only)
    exclude_outliers = st.sidebar.checkbox("Exclude outliers from average", value=False, key="exclude_outliers")

    # Chart series defaults: debits only by default
    show_debit = st.sidebar.checkbox("Show Debit (Total_Spent)", value=True, key="show_debit")
    show_credit = st.sidebar.checkbox("Show Credit (Total_Credit)", value=False, key="show_credit")

    # Chart type moved to sidebar
    chart_type = st.sidebar.selectbox("Chart type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=0)

    # Top-N for category chart (in sidebar)
    top_n = st.sidebar.slider("Top-N categories", 3, 20, 5, key="top_n")

    # Totals mode radio (move to sidebar, keep earlier behavior)
    totals_mode = st.sidebar.radio("Totals mode", ["Single date", "Date range"], index=0)

    # ------------------ Apply Filters ------------------
    filtered_df = converted_df[converted_df["Bank"].isin(sel_banks)].copy()

    # Note: chart / table filters use sel_banks and will still filter by month-range as before when applying start/end date windows.
    # (Monthly average is controlled by separate Year/Month selectors above.)

    # ------------------ Compute filtered daily totals used for charts ------------------
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

    # ------------------ Date Range (for rows & totals) ------------------
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

    # ------------------ Compute totals for selected date/range (for rows & summary metrics) ------------------
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

    # ------------------ Monthly average metric logic (uses separate Year + Month selectors) ------------------
    def _safe_mean(s):
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        return float(s2.mean()) if not s2.empty else None

    def _compute_month_avg_from_merged(merged_df, year, month, replace_outliers=False):
        """
        same IQR replacement logic as before
        """
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

    # Determine which month/year to compute the metric for based on sidebar controls
    metric_year = None
    metric_month = None
    if sel_year != "All" and sel_month_name != "All":
        try:
            metric_year = int(sel_year)
            # month name -> month number
            inv_map = {v: k for k, v in month_map.items()}
            metric_month = inv_map.get(sel_month_name)
        except Exception:
            metric_year = metric_month = None
    else:
        # fallback: pick latest month from merged_all
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

    # ------------------ Top-right compact metric (moved here with correct formatting) ------------------
    top_left, top_mid, top_right = st.columns([6, 2, 2])
    with top_right:
        # New, corrected title and display (moved from bottom)
        title_small = "Monthly average (Selected metric month)"
        # month label like "Sep-25"
        if metric_year and metric_month:
            month_label = pd.Timestamp(metric_year, metric_month, 1).strftime("%b-%y")
        else:
            month_label = "—"

        if metric_avg is None:
            metric_text = "N/A"
        else:
            metric_text = f"₹{metric_avg:,.2f}"

        # delta calculation
        if prev_avg is None or metric_avg is None:
            delta_html = f"<div style='text-align:right; color:gray; font-weight:600'>N/A</div>"
        else:
            diff = metric_avg - prev_avg
            try:
                if abs(prev_avg) > 1e-9:
                    pct = (diff / abs(prev_avg)) * 100.0
                    delta_label = f"{pct:+.1f}%"
                else:
                    # when previous is zero or nearly zero, show absolute diff with sign
                    delta_label = f"{diff:+.2f}"
            except Exception:
                delta_label = f"{diff:+.2f}"

            if diff > 0:
                color = "red"
                arrow = "▲"
            elif diff < 0:
                color = "green"
                arrow = "▼"
            else:
                color = "gray"
                arrow = "►"

            delta_html = f"<div style='text-align:right; color:{color}; font-weight:700'>{arrow} {delta_label}</div>"

        # Render: month label (small), amount (big), delta (colored)
        st.markdown(
            "<div style='text-align:right'>"
            f"<div style='font-size:12px;color:#666'>{title_small}</div>"
            f"<div style='font-size:14px;color:#444'>{month_label}</div>"
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

    # ------------------ Add New Row (date picker only, choose bank from data or Other) ------------------
    st.markdown("---")
    st.write("➕ Add a new transaction")

    def build_timestamp_str_using_now(chosen_date: date) -> (str, datetime):
        """
        Format datetime in 'MM/DD/YYYY HH:MM' using chosen date + current UTC time at submit.
        Returns (formatted_string, datetime_object).
        Example: '10/23/2025 18:26'
        """
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
                # Bank selection: pick from session_state bank list OR enter manual value
                add_bank_options = st.session_state["bank_options"].copy() if st.session_state.get("bank_options") else ["Unknown"]
                add_bank_options = sorted(list(set(add_bank_options)))
                if "Other (enter below)" not in add_bank_options:
                    add_bank_options.append("Other (enter below)")

                # Selectbox for bank
                chosen_bank_sel = st.selectbox("Bank", options=add_bank_options, index=0, key="add_bank_select")

                # Only show the custom bank input when Other is selected (fix for "can't enter new bank")
                bank_other = ""
                if chosen_bank_sel == "Other (enter below)":
                    bank_other = st.text_input("Bank (custom) — used when 'Other (enter below)' selected", value="", key="add_bank_other")

                txn_type = st.selectbox("Type", ["debit", "credit"])
                amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0, format="%.2f")
                msg = st.text_input("Message / Description", "")
                submit_add = st.form_submit_button("Save new row")

                if submit_add:
                    try:
                        # choose bank value (manual override if Other selected)
                        if chosen_bank_sel == "Other (enter below)":
                            chosen_bank = bank_other.strip() if bank_other and bank_other.strip() else "Unknown"
                        else:
                            chosen_bank = chosen_bank_sel if chosen_bank_sel else "Unknown"

                        timestamp_str, timestamp_dt = build_timestamp_str_using_now(new_date)

                        new_row = {
                            # DateTime string exactly as requested (MM/DD/YYYY HH:MM)
                            "DateTime": timestamp_str,
                            # timestamp object preserved for transform parsing
                            "timestamp": timestamp_dt,
                            # date cell written in same readable full format (optional)
                            "date": timestamp_dt.strftime("%m/%d/%Y %H:%M"),
                            "Bank": chosen_bank,
                            "Type": txn_type,
                            "Amount": amount,
                            "Message": msg,
                            "is_deleted": "false",
                        }

                        # Validate required fields minimally
                        if chosen_bank == "Unknown":
                            # let it still append if user explicitly chose Unknown, but prefer to prompt
                            st.info("Bank will be recorded as 'Unknown'. If you meant to add a custom name, fill the custom bank field and submit again.")
                        res = io_mod.append_new_row(
                            spreadsheet_id=SHEET_ID,
                            range_name=APPEND_RANGE,
                            new_row_dict=new_row,
                            creds_info=creds_info,
                            history_range=RANGE,
                        )

                        if res.get("status") == "ok":
                            # add the bank to session_state bank_options so it appears in filters immediately
                            if chosen_bank and chosen_bank not in st.session_state["bank_options"]:
                                st.session_state["bank_options"].append(chosen_bank)
                                st.session_state["bank_options"] = sorted(list(set(st.session_state["bank_options"])))
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

    # ------------------ Move Debits/Credits/Net metrics to bottom (after everything) ------------------
    st.markdown("---")
    st.subheader("Summary (Selected date/range)")
    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    c1.metric("Credits", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    c2.metric("Debits", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    c3.metric("Net (Credits − Debits)", f"₹{(credit_sum - debit_sum):,.0f}")

    # Note: the detailed monthly-average block was intentionally moved up to the top-right area.
    # The bottom copy has been removed to avoid duplication.

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Unexpected error in app. See traceback below.")
        st.exception(e)
        st.text(traceback.format_exc())
