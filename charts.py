# charts.py - visualization utilities (Daily line, Monthly bars, Top-N categories)
import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional, List

alt.data_transformers.disable_max_rows()  # safe for larger datasets; rely on server resources


def _ensure_date_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _is_deleted_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Detect an 'is_deleted' column (case-insensitive) and return a boolean mask (True for deleted rows).
    Returns None if no is_deleted column is present.
    Accepts boolean, numeric (1/0) and string values like 'true','t','1','yes'.
    """
    if df is None or df.empty:
        return None
    isdel_col = next((c for c in df.columns if str(c).lower() == 'is_deleted'), None)
    if isdel_col is None:
        return None
    s = df[isdel_col]
    try:
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(int) == 1
        lowered = s.astype(str).str.strip().str.lower().fillna('')
        return lowered.isin(['true', 't', '1', 'yes', 'y'])
    except Exception:
        try:
            lowered = s.astype(str).str.strip().str.lower().fillna('')
            return lowered.isin(['true', 't', '1', 'yes', 'y'])
        except Exception:
            return None


def _filter_out_deleted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df excluding rows marked deleted (if is_deleted column exists).
    Otherwise returns df.copy().
    """
    if df is None:
        return pd.DataFrame()
    mask = _is_deleted_mask(df)
    if mask is None:
        return df.copy()
    try:
        return df.loc[~mask].copy().reset_index(drop=True)
    except Exception:
        # defensive fallback
        try:
            return df.loc[~mask.fillna(False)].copy().reset_index(drop=True)
        except Exception:
            return df.copy()


def render_chart(plot_df: pd.DataFrame,
                 converted_df: pd.DataFrame,
                 chart_type: str,
                 series_selected: List[str],
                 top_n: int = 5,
                 height: int = 420) -> Optional[pd.Timestamp]:
    """
    Render a chart for aggregated data.
    - plot_df: aggregated daily DataFrame with Date, Total_Spent, Total_Credit
    - converted_df: cleaned transactions DataFrame (for Top-N and drilldown)
    - chart_type: "Daily line" | "Monthly bars" | "Top categories (Top-N)"
    - series_selected: which series to include (['Total_Spent','Total_Credit'])
    - top_n: for Top-N chart
    Returns:
      - None (no clicked-date currently). Client-side drilldown is handled inside the chart.
    """
    # defensively filter out deleted rows from inputs
    plot_df = _filter_out_deleted(plot_df) if plot_df is not None else pd.DataFrame()
    converted_df = _filter_out_deleted(converted_df) if converted_df is not None else pd.DataFrame()

    if plot_df is None or plot_df.empty:
        st.info("No aggregated data available for charting.")
        return None

    chart_type = (chart_type or "").strip()

    if chart_type == "Daily line":
        return _render_daily_line(plot_df, converted_df, series_selected, height)
    elif chart_type == "Monthly bars":
        return _render_monthly_bars(plot_df, series_selected, height)
    elif chart_type.startswith("Top"):
        return _render_top_categories(converted_df, top_n, height)
    else:
        st.error(f"Unknown chart type: {chart_type}")
        return None


# ------------------ Chart implementations ------------------ #
def _render_daily_line(plot_df: pd.DataFrame, converted_df: pd.DataFrame, series_selected: List[str], height: int):
    """
    Daily line chart with client-side drilldown:
      - top pane: daily lines/points (click a point to select a date)
      - bottom pane: transactions for the selected date (bar list sorted by amount)
    Selection is client-side via Altair selection; no server roundtrip.
    """
    df = _ensure_date_col(plot_df, "Date")
    vars_to_plot = [c for c in ['Total_Spent', 'Total_Credit'] if c in series_selected and c in df.columns]
    if not vars_to_plot:
        st.info("No series selected for plotting.")
        return None

    # Prepare long format for line chart
    long = df.melt(id_vars='Date', value_vars=vars_to_plot, var_name='Type', value_name='Amount').sort_values('Date')
    long['Amount'] = pd.to_numeric(long['Amount'], errors='coerce').fillna(0.0)

    # selection: user clicks a Date point (single selection)
    date_sel = alt.selection_single(
        fields=['Date'],
        nearest=True,
        on='click',
        empty='none',
        clear='dblclick'
    )

    # color scale (kept small & stable)
    color_scale = alt.Scale(domain=['Total_Spent', 'Total_Credit'], range=['#d62728', '#2ca02c'])

    base = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Amount:Q', title='Amount', axis=alt.Axis(format=",.0f")),
        color=alt.Color('Type:N', title='Type', scale=color_scale),
        tooltip=[
            alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('Amount:Q', title='Amount', format=',')
        ],
        opacity=alt.condition(date_sel, alt.value(1.0), alt.value(0.8))
    ).add_selection(date_sel).interactive()

    # If no converted_df provided or it lacks timestamp, just show the line chart
    if converted_df is None or converted_df.empty:
        st.altair_chart(base.properties(height=height), use_container_width=True)
        return None

    # Prepare transactions for detail pane
    tx = converted_df.copy()
    # ensure timestamp exists and normalize to date
    if 'timestamp' in tx.columns:
        tx['timestamp'] = pd.to_datetime(tx['timestamp'], errors='coerce')
    elif 'date' in tx.columns:
        tx['timestamp'] = pd.to_datetime(tx['date'], errors='coerce')
    else:
        tx['timestamp'] = pd.NaT
    tx['Date'] = pd.to_datetime(tx['timestamp'], errors='coerce').dt.normalize()
    tx['Date_str'] = tx['Date'].dt.strftime('%Y-%m-%d')
    tx['Amount_numeric'] = pd.to_numeric(tx.get('Amount', 0), errors='coerce').fillna(0.0)

    # Add a stable row id for ordering/display
    tx = tx.reset_index().rename(columns={'index': 'row_index'})

    # Compute per-date rank in pandas so we avoid transform_window schema issues
    try:
        tx['rank'] = tx.groupby('Date')['Amount_numeric'].rank(method='first', ascending=False)
        # make ordinal for Altair
        tx['rank'] = tx['rank'].fillna(999999).astype(int)
    except Exception:
        tx['rank'] = tx.reset_index().index.astype(int)

    # Lower chart: show transactions for selected date as horizontal bars sorted by Amount
    detail = alt.Chart(tx).transform_filter(
        date_sel
    ).mark_bar().encode(
        x=alt.X('Amount_numeric:Q', title='Amount', axis=alt.Axis(format=",.0f")),
        y=alt.Y('rank:O', title=None, axis=None),
        color=alt.Color('Type:N', scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip('Date_str:N', title='Date'),
            alt.Tooltip('Bank:N', title='Bank'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('Amount_numeric:Q', title='Amount', format=','),
            alt.Tooltip('Message:N', title='Message'),
            alt.Tooltip('_source_sheet:N', title='Source'),
            alt.Tooltip('_sheet_row_idx:N', title='Sheet row idx')
        ]
    ).properties(height=max(120, int(height * 0.35)))

    # Compose vconcat: top line chart + bottom details (bottom will be empty until a date is clicked)
    v = alt.vconcat(
        base.properties(height=max(200, int(height * 0.6))),
        detail
    ).resolve_scale(color='independent')

    st.altair_chart(v, use_container_width=True)
    return None


def _render_monthly_bars(plot_df: pd.DataFrame, series_selected: List[str], height: int):
    df = _ensure_date_col(plot_df, "Date")

    # Create Year-Month label
    df = df.copy()
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)  # e.g., "2025-10"
    # Decide which series to show: if both, show stacked bars; else single series
    vars_to_plot = [c for c in ['Total_Spent', 'Total_Credit'] if c in series_selected and c in df.columns]
    if not vars_to_plot:
        st.info("No series selected for plotting.")
        return None

    # aggregate by YearMonth
    agg = df.groupby('YearMonth')[vars_to_plot].sum().reset_index()
    # melt for stacked/grouped bars
    long = agg.melt(id_vars='YearMonth', value_vars=vars_to_plot, var_name='Type', value_name='Amount')
    long['Amount'] = pd.to_numeric(long['Amount'], errors='coerce').fillna(0.0)
    # preserve chronological order
    yearmonth_order = sorted(long['YearMonth'].unique(), key=lambda x: pd.to_datetime(x + "-01"))

    color_scale = alt.Scale(domain=['Total_Spent', 'Total_Credit'], range=['#d62728', '#2ca02c'])

    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X('YearMonth:N', sort=yearmonth_order, title='Month'),
        y=alt.Y('Amount:Q', title='Amount', axis=alt.Axis(format=",.0f")),
        color=alt.Color('Type:N', title='Type', scale=color_scale),
        tooltip=[
            alt.Tooltip('YearMonth:N', title='Month'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('Amount:Q', title='Amount', format=',')
        ]
    ).interactive()

    st.altair_chart(chart.properties(height=height), use_container_width=True)
    return None


def _render_top_categories(converted_df: pd.DataFrame, top_n: int, height: int):
    if converted_df is None or converted_df.empty:
        st.info("No transaction data available for Top-N categories.")
        return None

    df = converted_df.copy()
    # Ensure timestamp and Amount numeric
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    df['Amount_numeric'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0.0)

    # Determine category column preference
    preferred_cols = ['Category', 'Merchant', 'merchant', 'category', 'description']
    cat_col = None
    for c in preferred_cols:
        if c in df.columns:
            cat_col = c
            break

    if cat_col is None:
        st.info("Top-N chart needs a Category or Merchant column. If you have category data, rename it to 'Category'.")
        return None

    # Define spend as positive amounts (i.e., amounts > 0 or Type == 'debit')
    if 'Type' in df.columns:
        df['Type_norm'] = df['Type'].astype(str).str.lower().str.strip()
        spend_mask = df['Type_norm'] == 'debit'
    else:
        spend_mask = df['Amount_numeric'] > 0

    spend_df = df[spend_mask].copy()
    if spend_df.empty:
        st.info("No spend (debit) transactions found for Top-N in the dataset.")
        return None

    agg = spend_df.groupby(cat_col)['Amount_numeric'].sum().reset_index().rename(columns={cat_col: 'Category', 'Amount_numeric': 'Total'})
    # Show top N by total spend
    agg = agg.sort_values('Total', ascending=False).head(top_n)
    agg['Total'] = agg['Total'].astype(float)

    chart = alt.Chart(agg).mark_bar().encode(
        x=alt.X('Total:Q', title='Total spend', axis=alt.Axis(format=",.0f")),
        y=alt.Y('Category:N', sort='-x', title='Category'),
        tooltip=[alt.Tooltip('Category:N', title='Category'), alt.Tooltip('Total:Q', title='Total', format=',')]
    ).properties(height=max(200, min(600, 40 * len(agg))))

    st.altair_chart(chart, use_container_width=True)
    return None
