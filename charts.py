# charts.py - visualization utilities (Daily line, Monthly bars, Top-N categories)

import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional, List

# Allow large datasets
alt.data_transformers.disable_max_rows()


# ------------------ Utility functions ------------------ #
def _ensure_date_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """Ensure the given column is in datetime format."""
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _is_deleted_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Detect 'is_deleted' column and return a boolean mask (True = deleted)."""
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
        return None


def _filter_out_deleted(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df excluding rows marked deleted."""
    if df is None:
        return pd.DataFrame()
    mask = _is_deleted_mask(df)
    if mask is None:
        return df.copy()
    return df.loc[~mask].copy().reset_index(drop=True)


# ------------------ Main chart entry ------------------ #
def render_chart(
    plot_df: pd.DataFrame,
    converted_df: pd.DataFrame,
    chart_type: str,
    series_selected: List[str],
    top_n: int = 5,
    height: int = 420
) -> None:
    """
    Render a chart based on chart_type:
      - "Daily line" : Daily time-series of spend/credit
      - "Monthly bars" : Monthly aggregated bar chart
      - "Top categories (Top-N)" : Top spending categories
    """
    plot_df = _filter_out_deleted(plot_df)
    converted_df = _filter_out_deleted(converted_df)

    if plot_df is None or plot_df.empty:
        st.info("No aggregated data available for charting.")
        return

    chart_type = (chart_type or "").strip()

    if chart_type == "Daily line":
        _render_daily_line(plot_df, converted_df, series_selected, height)
    elif chart_type == "Monthly bars":
        _render_monthly_bars(plot_df, series_selected, height)
    elif chart_type.startswith("Top"):
        _render_top_categories(converted_df, top_n, height)
    else:
        st.error(f"Unknown chart type: {chart_type}")


# ------------------ Daily Line Chart ------------------ #
def _render_daily_line(plot_df: pd.DataFrame, converted_df: pd.DataFrame, series_selected: List[str], height: int):
    """Render interactive daily line chart with transaction drilldown."""
    df = _ensure_date_col(plot_df, "Date")
    vars_to_plot = [c for c in ['Total_Spent', 'Total_Credit'] if c in series_selected and c in df.columns]

    if not vars_to_plot:
        st.info("No series selected for plotting.")
        return

    long = df.melt(id_vars='Date', value_vars=vars_to_plot, var_name='Type', value_name='Amount').sort_values('Date')
    long['Amount'] = pd.to_numeric(long['Amount'], errors='coerce').fillna(0.0)

    date_sel = alt.selection_single(fields=['Date'], nearest=True, on='click', empty='none', clear='dblclick')
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

    # If no transaction data, just show the top chart
    if converted_df is None or converted_df.empty:
        st.altair_chart(base.properties(height=height), use_container_width=True)
        return

    # Prepare drilldown data
    tx = converted_df.copy()
    if 'timestamp' in tx.columns:
        tx['timestamp'] = pd.to_datetime(tx['timestamp'], errors='coerce')
    elif 'date' in tx.columns:
        tx['timestamp'] = pd.to_datetime(tx['date'], errors='coerce')
    tx['Date'] = tx['timestamp'].dt.normalize()
    tx['Amount_numeric'] = pd.to_numeric(tx.get('Amount', 0), errors='coerce').fillna(0.0)

    tx = tx.reset_index().rename(columns={'index': 'row_index'})
    tx['rank'] = tx.groupby('Date')['Amount_numeric'].rank(method='first', ascending=False).fillna(999999).astype(int)

    detail = alt.Chart(tx).transform_filter(date_sel).mark_bar().encode(
        x=alt.X('Amount_numeric:Q', title='Amount', axis=alt.Axis(format=",.0f")),
        y=alt.Y('rank:O', title=None, axis=None),
        color=alt.Color('Type:N', scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip('Date:T', title='Date'),
            alt.Tooltip('Bank:N', title='Bank'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('Amount_numeric:Q', title='Amount', format=','),
            alt.Tooltip('Message:N', title='Message'),
        ]
    ).properties(height=max(120, int(height * 0.35)))

    combined = alt.vconcat(
        base.properties(height=max(200, int(height * 0.6))),
        detail
    ).resolve_scale(color='independent')

    st.altair_chart(combined, use_container_width=True)


# ------------------ Monthly Bars Chart ------------------ #
def _render_monthly_bars(plot_df: pd.DataFrame, series_selected: List[str], height: int):
    """Aggregate by month and plot bars for Total_Spent / Total_Credit."""
    df = _ensure_date_col(plot_df, "Date")
    if df.empty:
        st.info("No data for Monthly Bars chart.")
        return

    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    vars_to_plot = [c for c in ['Total_Spent', 'Total_Credit'] if c in series_selected and c in df.columns]

    if not vars_to_plot:
        st.info("No valid series selected for Monthly Bars chart.")
        return

    agg = df.groupby('YearMonth')[vars_to_plot].sum().reset_index()
    long = agg.melt(id_vars='YearMonth', value_vars=vars_to_plot, var_name='Type', value_name='Amount')
    long['Amount'] = pd.to_numeric(long['Amount'], errors='coerce').fillna(0.0)
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
    ).properties(height=height).interactive()

    st.altair_chart(chart, use_container_width=True)


# ------------------ Top-N Categories Chart ------------------ #
def _render_top_categories(converted_df: pd.DataFrame, top_n: int, height: int):
    """Show top N spending categories."""
    if converted_df is None or converted_df.empty:
        st.info("No transaction data available for Top-N categories.")
        return

    df = converted_df.copy()
    df['Amount_numeric'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0.0)

    # Choose category column
    preferred_cols = ['Category', 'Merchant', 'merchant', 'category', 'description']
    cat_col = next((c for c in preferred_cols if c in df.columns), None)

    if cat_col is None:
        st.info("Top-N chart needs a Category or Merchant column.")
        return

    # Spending filter
    if 'Type' in df.columns:
        df['Type_norm'] = df['Type'].astype(str).str.lower().str.strip()
        spend_mask = df['Type_norm'] == 'debit'
    else:
        spend_mask = df['Amount_numeric'] > 0

    spend_df = df[spend_mask].copy()
    if spend_df.empty:
        st.info("No debit/spend transactions found for Top-N chart.")
        return

    agg = (
        spend_df.groupby(cat_col)['Amount_numeric']
        .sum()
        .reset_index()
        .rename(columns={cat_col: 'Category', 'Amount_numeric': 'Total'})
    )

    agg = agg.sort_values('Total', ascending=False).head(top_n)
    agg['Total'] = agg['Total'].astype(float)

    chart = alt.Chart(agg).mark_bar().encode(
        x=alt.X('Total:Q', title='Total spend', axis=alt.Axis(format=",.0f")),
        y=alt.Y('Category:N', sort='-x', title='Category'),
        tooltip=[
            alt.Tooltip('Category:N', title='Category'),
            alt.Tooltip('Total:Q', title='Total', format=',')
        ]
    ).properties(height=max(200, min(600, 40 * len(agg))))

    st.altair_chart(chart, use_container_width=True)
