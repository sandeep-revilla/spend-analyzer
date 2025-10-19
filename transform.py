# transform.py  -- pure data transformation utilities
import pandas as pd
from typing import Optional

def _is_date_like_column(col_name: str) -> bool:
    lname = str(col_name).lower()
    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'txn']
    return any(k in lname for k in date_keywords) or lname.startswith("unnamed")

def _is_amount_like_column(col_name: str) -> bool:
    lname = str(col_name).lower()
    num_keywords = ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit', 'spent']
    return any(k in lname for k in num_keywords)

def _detect_is_deleted_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    If an 'is_deleted' column (case-insensitive) exists in df, return a boolean mask
    where True indicates the row is deleted. Accepts boolean, numeric, and common string
    values ('true','t','1','yes'). Returns None if no such column exists.
    """
    isdel_col = next((c for c in df.columns if str(c).lower() == 'is_deleted'), None)
    if isdel_col is None:
        return None

    s = df[isdel_col]
    try:
        # boolean dtype
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool)
        # numeric (1/0)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(int) == 1
        # strings / mixed - normalize and check common true tokens
        lowered = s.astype(str).str.strip().str.lower().fillna('')
        return lowered.isin(['true', 't', '1', 'yes', 'y'])
    except Exception:
        try:
            lowered = s.astype(str).str.strip().str.lower().fillna('')
            return lowered.isin(['true', 't', '1', 'yes', 'y'])
        except Exception:
            return None

def _prefer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Prefer an explicit 'DateTime' (case-insensitive) or any column containing 'datetime'
    if present. Otherwise return None (caller will fallback to heuristic detection).
    """
    for c in df.columns:
        if str(c).lower() == 'datetime' or str(c).lower() == 'date_time':
            return c
    for c in df.columns:
        if 'datetime' in str(c).lower():
            return c
    return None

def convert_columns_and_derives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns:
      - remove rows marked deleted via an 'is_deleted' column (case-insensitive)
      - detect and coerce date/time columns -> timestamp (preferring DateTime if present)
      - detect and coerce numeric columns -> Amount (float)
      - create 'date' column (date part of timestamp)
      - infer Type ('debit'/'credit') if missing based on sign of Amount

    Returns a cleaned DataFrame (does not mutate input). Preserves any extra columns
    (like _sheet_row_idx, _source_sheet) so they survive transformation.
    Rows where Type is unknown are dropped (same behavior as before).
    """
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    # Work on a copy to avoid mutating caller's DataFrame
    df = df.copy()

    # Preserve original column names (strip only whitespace)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # 0) Filter out soft-deleted rows early (if present)
    try:
        isdel_mask = _detect_is_deleted_mask(df)
        if isdel_mask is not None:
            df = df.loc[~isdel_mask].copy().reset_index(drop=True)
    except Exception:
        # if detection fails, continue without filtering
        pass

    # 1) Prefer explicit DateTime-like column if available
    primary_dt_col = _prefer_datetime_column(df)

    # 2) If not found, fall back to the original heuristic detection
    if primary_dt_col is None:
        for col in df.columns:
            try:
                if _is_date_like_column(col):
                    parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                    if parsed.notna().sum() > 0:
                        primary_dt_col = col
                        df[col] = parsed
                        break
            except Exception:
                continue

    # fallback: try to find any object column that can be parsed as datetime
    if primary_dt_col is None:
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                    if parsed.notna().sum() >= 3:
                        primary_dt_col = col
                        df[col] = parsed
                        break
            except Exception:
                continue

    # 3) Detect amount-like columns and coerce to numeric (choose preferred)
    amount_cols = []
    for col in df.columns:
        try:
            if _is_amount_like_column(col):
                coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                df[col] = coerced
                amount_cols.append(col)
        except Exception:
            continue

    # Heuristic: if no obvious amount column found, try to coerce any object column that looks numeric
    if not amount_cols:
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
                    parsed = pd.to_numeric(sample, errors='coerce')
                    if parsed.notna().sum() >= 3:
                        coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                        df[col] = coerced
                        amount_cols.append(col)
                        break
            except Exception:
                continue

    # Choose preferred amount column name (case-insensitive matching)
    preferred = None
    candidates = ['amount','total_spent','totalspent','total','txn amount','value','spent','amt']
    for candidate in candidates:
        for col in df.columns:
            if str(col).lower() == candidate:
                preferred = col
                break
        if preferred:
            break
    if not preferred and amount_cols:
        preferred = amount_cols[0]

    if preferred:
        # create canonical 'Amount' column (unless it already exists and is numeric)
        if preferred != 'Amount':
            if 'Amount' not in df.columns:
                try:
                    df.rename(columns={preferred: 'Amount'}, inplace=True)
                except Exception:
                    df['Amount'] = pd.to_numeric(df[preferred], errors='coerce')
        else:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    else:
        if 'Amount' not in df.columns:
            df['Amount'] = pd.NA
        else:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # 4) Create canonical timestamp and date columns (prefer primary_dt_col)
    try:
        if primary_dt_col:
            # parse preserving time component
            df['timestamp'] = pd.to_datetime(df[primary_dt_col], errors='coerce')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['timestamp'] = pd.NaT
    except Exception:
        df['timestamp'] = pd.NaT

    # date (date part) - keep as date objects (not datetime) to ease grouping
    try:
        df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    except Exception:
        df['date'] = pd.NA

    # 5) Infer or normalize Type (case-insensitive detection of existing 'type' column)
    existing_type_col = next((c for c in df.columns if str(c).lower() == 'type'), None)

    if existing_type_col is None:
        try:
            df['Type'] = pd.NA
            mask_pos = df['Amount'].notna() & (df['Amount'] > 0)
            mask_neg = df['Amount'].notna() & (df['Amount'] < 0)
            df.loc[mask_pos, 'Type'] = 'debit'
            df.loc[mask_neg, 'Type'] = 'credit'
        except Exception:
            df['Type'] = pd.NA
    else:
        try:
            df['Type'] = df[existing_type_col].astype(str).str.lower().str.strip()
        except Exception:
            df['Type'] = df[existing_type_col].astype(str)

    # Normalize Type column to canonical lower-case strings and mark empties as 'unknown'
    try:
        df['Type'] = df['Type'].astype(str).str.lower().str.strip()
        df['Type'] = df['Type'].replace({'nan': 'unknown', 'none': 'unknown', '': 'unknown', 'na': 'unknown', 'null': 'unknown'})
    except Exception:
        df['Type'] = 'unknown'

    # DROP rows where Type is unknown (preserve other columns)
    try:
        df = df[df['Type'] != 'unknown'].copy().reset_index(drop=True)
    except Exception:
        pass

    # final reorder: put timestamp and date at front but preserve other columns (including _sheet_row_idx, is_deleted etc.)
    cols = list(df.columns)
    final = []
    if 'timestamp' in cols:
        final.append('timestamp')
    if 'date' in cols:
        final.append('date')
    for c in cols:
        if c not in final:
            final.append(c)
    df = df[final]

    return df

def compute_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily totals DataFrame with columns:
      - Date (normalized datetime)
      - Total_Spent (float)  <- sum of debits / spend
      - Total_Credit (float) <- sum of credits
    Logic:
      - If 'Type' column exists and contains debit/credit: use it.
      - Else: treat positive Amount as spend (debit), negative as credit.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

    w = df.copy()

    # determine grouping date
    if 'date' in w.columns and w['date'].notna().any():
        grp = pd.to_datetime(w['date']).dt.normalize()
    elif 'timestamp' in w.columns and w['timestamp'].notna().any():
        grp = pd.to_datetime(w['timestamp']).dt.normalize()
    else:
        # try to find any datetime-like column
        found = None
        for c in w.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(w[c]) and w[c].notna().any():
                    found = c
                    break
            except Exception:
                continue
        if found:
            grp = pd.to_datetime(w[found]).dt.normalize()
        else:
            return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

    w['_group_date'] = grp
    w['Amount_numeric'] = pd.to_numeric(w.get('Amount', 0), errors='coerce').fillna(0.0).astype('float64')

    # If Type present and has meaningful values, use it
    if 'Type' in w.columns and w['Type'].astype(str).str.strip().any():
        w['Type_norm'] = w['Type'].astype(str).str.lower().str.strip()
        debit_df = w[w['Type_norm'] == 'debit']
        credit_df = w[w['Type_norm'] == 'credit']
        daily_spend = debit_df.groupby(debit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Spent'})
        daily_credit = credit_df.groupby(credit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Credit'})
    else:
        # fallback: positive = spend, negative = credit
        debit_df = w[w['Amount_numeric'] > 0]
        credit_df = w[w['Amount_numeric'] < 0]
        daily_spend = debit_df.groupby(debit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Spent'})
        daily_credit = credit_df.groupby(credit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Credit'})
        # credit sums are negative values; make them positive for reporting
        if 'Total_Credit' in daily_credit.columns:
            daily_credit['Total_Credit'] = daily_credit['Total_Credit'].abs()

    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    merged['Total_Spent'] = merged.get('Total_Spent', 0).astype('float64')
    merged['Total_Credit'] = merged.get('Total_Credit', 0).astype('float64')
    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged
