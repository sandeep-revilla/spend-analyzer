# io_helpers.py -- data I/O helpers (pure functions, no Streamlit runtime objects passed into cached functions)
import json
import os
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

# Optional Google Sheets imports (will raise only when used)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:
    service_account = None
    build = None
    HttpError = Exception


def parse_service_account_secret(raw: Any) -> Dict:
    """Parse a service account JSON blob stored as dict or string. Returns a plain dict."""
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace('\\n', '\n'))
        except Exception:
            return json.loads(s.replace('\n', '\\n'))


def _normalize_rows(values: List[List[str]]) -> Tuple[List[str], List[List]]:
    """
    Convert Google Sheets 'values' (list of rows) into header + normalized rows.
    If first row looks like a header, use it; otherwise synthesize col_{i} headers.
    """
    if not values:
        return [], []
    header_row = [str(x).strip() for x in values[0]]
    if all((h == "" or h.lower().startswith(("unnamed", "column", "nan"))) for h in header_row):
        max_cols = max(len(r) for r in values)
        header = [f"col_{i}" for i in range(max_cols)]
        data_rows = values
    else:
        header = header_row
        data_rows = values[1:]
    col_count = len(header)
    normalized = []
    for r in data_rows:
        if len(r) < col_count:
            r = r + [None] * (col_count - len(r))
        elif len(r) > col_count:
            r = r[:col_count]
        normalized.append(r)
    return header, normalized


def values_to_dataframe(values: List[List[str]]) -> pd.DataFrame:
    """Turn a Google Sheets 'values' payload into a pandas DataFrame."""
    if not values:
        return pd.DataFrame()
    header, rows = _normalize_rows(values)
    try:
        return pd.DataFrame(rows, columns=header)
    except Exception:
        df = pd.DataFrame(rows)
        if header and df.shape[1] == len(header):
            df.columns = header
        return df


def build_sheets_service_from_info(creds_info: Dict):
    """Create Google Sheets API service from service-account info dict (read-only scope)."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def build_sheets_service_from_file(creds_file: str):
    """Create Google Sheets API service from local service-account JSON file path (read-only scope)."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def build_sheets_service_write_from_info(creds_info: Dict):
    """Create Google Sheets API service from service-account info dict (write access)."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def build_sheets_service_write_from_file(creds_file: str):
    """Create Google Sheets API service from local service-account JSON file path (write access)."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


@st.cache_data(ttl=600)
def read_google_sheet(spreadsheet_id: str, range_name: str,
                      creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
    """
    Read a Google Sheet and return a pandas DataFrame.
    - creds_info: parsed JSON dict for service account (plain dict) OR None
    - creds_file: path to service account JSON on disk OR None

    IMPORTANT: Do NOT pass Streamlit runtime objects (e.g., st.secrets) to this function.
    Pass a plain dict for creds_info (use parse_service_account_secret to convert).
    This function is cached by Streamlit (ttl default 600 seconds).
    """
    # prefer explicit creds_info / creds_file; else raise
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        raise ValueError("No credentials found. Provide creds_info (plain dict) or a valid creds_file path.")

    service = None
    if creds_info is not None:
        service = build_sheets_service_from_info(creds_info)
    else:
        service = build_sheets_service_from_file(creds_file)

    try:
        sheet = service.spreadsheets()
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        raise RuntimeError(f"Google Sheets API error: {e}")
    return values_to_dataframe(values)


# ---------------------------------------------------------------------
# New write helpers (ensure headers sync between history + append, safe append)
# ---------------------------------------------------------------------


def _get_write_service(creds_info: Optional[Dict] = None, creds_file: Optional[str] = None):
    """
    Return a Google Sheets service with write access.
    Prefers creds_info (plain dict). If not provided, uses creds_file path.
    """
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        raise ValueError("No credentials found for write operation. Provide creds_info or a valid creds_file path.")
    if creds_info is not None:
        return build_sheets_service_write_from_info(creds_info)
    else:
        return build_sheets_service_write_from_file(creds_file)


def ensure_sheet_headers_match(spreadsheet_id: str,
                               history_range: Optional[str],
                               append_range: str,
                               creds_info: Optional[Dict] = None,
                               creds_file: Optional[str] = None) -> Dict:
    """
    Ensure the Append sheet header contains all columns present in the History sheet header.
    - If append sheet is missing columns from history, add them to the end of the append header
      and extend existing data rows with empty cells so columns align.
    - If history_range is None, this becomes a no-op (returns ok) — useful when only append header is known.
    Returns dict {status: 'ok'|'error', 'added': [cols], 'message': str}
    """
    service = _get_write_service(creds_info, creds_file)
    sheet = service.spreadsheets()

    # Read history header
    history_header = []
    if history_range:
        try:
            res_h = sheet.values().get(spreadsheetId=spreadsheet_id, range=history_range).execute()
            vals_h = res_h.get("values", [])
            if vals_h:
                history_header = [str(x).strip() for x in vals_h[0]]
        except HttpError as e:
            return {"status": "error", "message": f"Failed to read history sheet header: {e}"}

    # Read append sheet header + data rows
    try:
        res_a = sheet.values().get(spreadsheetId=spreadsheet_id, range=append_range).execute()
        vals_a = res_a.get("values", [])
    except HttpError as e:
        return {"status": "error", "message": f"Failed to read append sheet: {e}"}

    append_header, append_rows = _normalize_rows(vals_a)

    # If append is empty and history header exists, create append header from history header
    if (not append_header or all((h == "" for h in append_header))) and history_header:
        append_header = history_header.copy()
        # write header and leave no data rows
        try:
            sheet.values().update(
                spreadsheetId=spreadsheet_id,
                range=append_range,
                valueInputOption='USER_ENTERED',
                body={'values': [append_header]}
            ).execute()
            return {"status": "ok", "added": append_header, "message": "Append header created from history."}
        except HttpError as e:
            return {"status": "error", "message": f"Failed to create append header: {e}"}

    # If history header is empty or not provided, nothing to sync
    if not history_header:
        return {"status": "ok", "added": [], "message": "No history header provided; nothing to add."}

    # Compute which history columns are missing in append (case-insensitive)
    lower_append = [h.lower() for h in append_header]
    missing = [h for h in history_header if h.lower() not in lower_append]

    if not missing:
        return {"status": "ok", "added": [], "message": "Append header already contains all history columns."}

    # Extend append_header and pad existing append rows with None for new columns
    new_header = append_header + missing
    for i in range(len(append_rows)):
        append_rows[i] = append_rows[i] + [None] * len(missing)

    # Build values (header + data rows) and write back to append_range (replace)
    out_values = [new_header] + append_rows
    try:
        sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range=append_range,
            valueInputOption='USER_ENTERED',
            body={'values': out_values}
        ).execute()
    except HttpError as e:
        return {"status": "error", "message": f"Failed to update append header: {e}"}

    return {"status": "ok", "added": missing, "message": f"Added {len(missing)} columns to append header."}


def mark_rows_deleted(spreadsheet_id: str, range_name: str,
                      creds_info: Optional[Dict] = None, creds_file: Optional[str] = None,
                      row_indices: Optional[List[int]] = None) -> Dict:
    """
    Soft-delete rows by setting / creating an 'is_deleted' column and marking the specified
    rows as TRUE.

    - spreadsheet_id, range_name: passed to Sheets API (range_name commonly a sheet name like 'History Transactions' or 'Append Transactions').
    - creds_info: plain dict from parse_service_account_secret OR creds_file: path on disk.
    - row_indices: list of integers (0-based) referring to the data rows (first data row after header is index 0).
      If None or empty -> nothing is changed.

    Returns: dict with status and count of rows updated.
    """
    if not row_indices:
        return {"status": "no-op", "updated": 0, "message": "No row indices provided."}

    service = _get_write_service(creds_info, creds_file)
    sheet = service.spreadsheets()

    try:
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        return {"status": "error", "message": f"Failed to read sheet: {e}"}

    # Normalize header + data rows
    header, data_rows = _normalize_rows(values)
    # If there was no header (empty sheet) treat as error
    if not header:
        return {"status": "error", "message": "Sheet appears empty, cannot mark rows deleted."}

    # Ensure 'is_deleted' column exists (case-insensitive)
    if 'is_deleted' not in [h.lower() for h in header]:
        header.append('is_deleted')
        for i in range(len(data_rows)):
            data_rows[i].append(None)
        is_deleted_idx = len(header) - 1
    else:
        # find index case-insensitively
        is_deleted_idx = next(i for i, h in enumerate(header) if h.lower() == 'is_deleted')

    updated = 0
    for idx in row_indices:
        if 0 <= idx < len(data_rows):
            # mark as TRUE string (consistent with user's string style)
            data_rows[idx][is_deleted_idx] = 'TRUE'
            updated += 1

    # Build values to write back (header row + data rows)
    out_values = [header] + data_rows

    # Use update to replace the range content
    try:
        sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body={'values': out_values}
        ).execute()
    except HttpError as e:
        return {"status": "error", "message": f"Failed to write sheet: {e}"}

    return {"status": "ok", "updated": updated}


def append_new_row(spreadsheet_id: str, range_name: str, new_row_dict: Dict[str, Any],
                   creds_info: Optional[Dict] = None, creds_file: Optional[str] = None,
                   history_range: Optional[str] = None) -> Dict:
    """
    Append a new row to the sheet (Append sheet). Optionally synchronize headers with History sheet
    before appending by providing history_range (sheet/tab name or A1-range for history header).

    - spreadsheet_id: spreadsheet id (same spreadsheet that holds history + append or the target spreadsheet)
    - range_name: append sheet name/range (e.g., 'Append Transactions')
    - new_row_dict: mapping of column name -> value for the new row. Keys are matched to existing header names
      case-insensitively. Missing header columns will be left blank.
    - history_range: optional history sheet range/name to sync headers from prior to append.
    - Returns a dict with status and the appended row number when available.

    NOTE: This function will not modify non-mentioned columns; they remain blank.
    """
    service = _get_write_service(creds_info, creds_file)
    sheet = service.spreadsheets()

    # First, optionally ensure headers match (synchronize append header to include history columns)
    if history_range:
        try:
            ensure_res = ensure_sheet_headers_match(spreadsheet_id=spreadsheet_id,
                                                   history_range=history_range,
                                                   append_range=range_name,
                                                   creds_info=creds_info,
                                                   creds_file=creds_file)
            if ensure_res.get('status') != 'ok':
                # proceed but warn; return error might be too strict — return error to caller
                return {"status": "error", "message": f"Failed to ensure headers match: {ensure_res.get('message')}"}
        except Exception as e:
            return {"status": "error", "message": f"Error while ensuring headers: {e}"}

    # Read current append sheet content to get final header
    try:
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        return {"status": "error", "message": f"Failed to read sheet before append: {e}"}

    header, data_rows = _normalize_rows(values)

    # If still no header (empty sheet) create header from new_row_dict keys (in insertion order)
    if not header:
        header = list(new_row_dict.keys())
        data_rows = []
        try:
            sheet.values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',
                body={'values': [header]}
            ).execute()
        except HttpError as e:
            return {"status": "error", "message": f"Failed to write header to empty append sheet: {e}"}

    # Ensure 'is_deleted' present in header; if not, append it so we always write the flag
    if 'is_deleted' not in [h.lower() for h in header]:
        header.append('is_deleted')
        for i in range(len(data_rows)):
            data_rows[i].append(None)

        # write back header + padded rows to persist header change before append
        try:
            sheet.values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',
                body={'values': [header] + data_rows}
            ).execute()
        except HttpError as e:
            return {"status": "error", "message": f"Failed to add is_deleted column to append sheet: {e}"}

    # Build row in header order (case-insensitive matching)
    row_out = []
    for col in header:
        # match keys case-sensitively; if not present try case-insensitive fallback
        if col in new_row_dict:
            v = new_row_dict[col]
        else:
            found = None
            for k in new_row_dict:
                if k.lower() == col.lower():
                    found = new_row_dict[k]
                    break
            v = found if found is not None else None

        # If the is_deleted column is present but user didn't provide it, default to string 'false'
        if str(col).lower() == 'is_deleted' and (v is None):
            v = 'false'

        # Convert pandas/numpy types or datetimes to strings so Sheets accepts them cleanly
        if v is None:
            row_out.append(None)
        else:
            # if pandas Timestamp / datetime -> convert to ISO (but prefer space-separated datetime)
            try:
                import pandas as _pd
                if isinstance(v, (_pd.Timestamp,)):
                    # convert to "YYYY-MM-DD HH:MM:SS" if possible
                    try:
                        row_out.append(v.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        row_out.append(str(v))
                    continue
            except Exception:
                pass
            try:
                from datetime import datetime, date
                if isinstance(v, (datetime, date)):
                    try:
                        if isinstance(v, date) and not isinstance(v, datetime):
                            # date only -> keep YYYY-MM-DD (but user likely provided datetime)
                            row_out.append(v.isoformat())
                        else:
                            row_out.append(v.strftime("%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        row_out.append(v.isoformat())
                    continue
            except Exception:
                pass
            row_out.append(v)

    # Append using Sheets API append
    try:
        append_res = sheet.values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': [row_out]}
        ).execute()
    except HttpError as e:
        return {"status": "error", "message": f"Failed to append row: {e}"}

    # Attempt to extract updatedRange / tableRange info for the appended row index (best-effort)
    try:
        updated_range = append_res.get('updates', {}).get('updatedRange')
        appended_row_number = None
        if updated_range and '!' in updated_range:
            rng = updated_range.split('!')[1]
            if ':' in rng:
                last = rng.split(':')[-1]
                import re
                m = re.search(r'(\d+)', last)
                if m:
                    appended_row_number = int(m.group(1))
    except Exception:
        appended_row_number = None

    return {"status": "ok", "appended_row_number": appended_row_number, "append_response": append_res}
