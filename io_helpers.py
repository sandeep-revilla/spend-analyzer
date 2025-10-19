# io_helpers.py -- data I/O helpers (pure functions, no Streamlit runtime objects passed in)
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

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
    # if first row looks like "Unnamed" header or blank, synthesize
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
        df = pd.DataFrame(rows, columns=header)
    except Exception:
        df = pd.DataFrame(rows)
        if header and df.shape[1] == len(header):
            df.columns = header
    # tidy column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _truthy_is_deleted(val: Any) -> bool:
    """Interpret a variety of string/number values as deleted True/False."""
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("true", "t", "1", "yes", "y")


def _normalize_is_deleted_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'is_deleted' exists (case-insensitive), rename to canonical 'is_deleted' and
    canonicalize values to uppercase 'TRUE'/'FALSE' strings for stable round-trip writes.
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    if "is_deleted" in lower_map:
        orig = lower_map["is_deleted"]
        if orig != "is_deleted":
            df = df.rename(columns={orig: "is_deleted"})
        df["is_deleted"] = df["is_deleted"].apply(lambda v: "TRUE" if _truthy_is_deleted(v) else "FALSE")
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


def read_google_sheet(spreadsheet_id: str, range_name: str,
                      creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
    """
    Read a Google Sheet and return a pandas DataFrame.
    - creds_info: parsed JSON dict for service account (plain dict) OR None
    - creds_file: path to service account JSON on disk OR None

    IMPORTANT: Do NOT pass Streamlit runtime objects to this function.
    This function does NOT perform caching; callers may implement caching if desired.

    The returned DataFrame will have normalized column names and, if an is_deleted column
    exists, it will be canonicalized to 'TRUE'/'FALSE' strings stored in the 'is_deleted' column.
    """
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        raise ValueError("No credentials found. Provide creds_info (plain dict) or a valid creds_file path.")

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

    df = values_to_dataframe(values)
    df = _normalize_is_deleted_column(df)
    return df


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
    - If history_range is None, this becomes a no-op (returns ok).
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

    if not history_header:
        return {"status": "ok", "added": [], "message": "No history header provided; nothing to add."}

    lower_append = [h.lower() for h in append_header]
    missing = [h for h in history_header if h.lower() not in lower_append]

    if not missing:
        return {"status": "ok", "added": [], "message": "Append header already contains all history columns."}

    new_header = append_header + missing
    for i in range(len(append_rows)):
        append_rows[i] = append_rows[i] + [None] * len(missing)

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

    header, data_rows = _normalize_rows(values)
    if not header:
        return {"status": "error", "message": "Sheet appears empty, cannot mark rows deleted."}

    lower_header = [h.lower() for h in header]
    if "is_deleted" not in lower_header:
        header.append("is_deleted")
        is_deleted_idx = len(header) - 1
        for i in range(len(data_rows)):
            data_rows[i].append(None)
    else:
        is_deleted_idx = next(i for i, h in enumerate(header) if h.lower() == "is_deleted")

    updated = 0
    for idx in row_indices:
        if isinstance(idx, int) and 0 <= idx < len(data_rows):
            data_rows[idx][is_deleted_idx] = "TRUE"
            updated += 1

    out_values = [header] + data_rows
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
    """
    service = _get_write_service(creds_info, creds_file)
    sheet = service.spreadsheets()

    # Optionally ensure headers match (synchronize append header to include history columns)
    if history_range:
        try:
            ensure_res = ensure_sheet_headers_match(spreadsheet_id=spreadsheet_id,
                                                   history_range=history_range,
                                                   append_range=range_name,
                                                   creds_info=creds_info,
                                                   creds_file=creds_file)
            if ensure_res.get("status") != "ok":
                return {"status": "error", "message": f"Failed to ensure headers match: {ensure_res.get('message')}"}
        except Exception as e:
            return {"status": "error", "message": f"Error while ensuring headers: {e}"}

    try:
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        return {"status": "error", "message": f"Failed to read sheet before append: {e}"}

    header, data_rows = _normalize_rows(values)

    if not header:
        header = list(new_row_dict.keys())
        data_rows = []
        try:
            sheet.values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption="USER_ENTERED",
                body={"values": [header]}
            ).execute()
        except HttpError as e:
            return {"status": "error", "message": f"Failed to write header to empty append sheet: {e}"}

    lower_header = [h.lower() for h in header]
    if "is_deleted" not in lower_header:
        header.append("is_deleted")
        for i in range(len(data_rows)):
            data_rows[i].append(None)
        # persist header change
        try:
            sheet.values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption="USER_ENTERED",
                body={"values": [header] + data_rows}
            ).execute()
        except HttpError as e:
            return {"status": "error", "message": f"Failed to add is_deleted column to append sheet: {e}"}

    # Build row in header order (case-insensitive matching)
    row_out: List[Any] = []
    for col in header:
        if col in new_row_dict:
            v = new_row_dict[col]
        else:
            # case-insensitive fallback
            found = None
            for k in new_row_dict:
                if k.lower() == col.lower():
                    found = new_row_dict[k]
                    break
            v = found if found is not None else None

        if str(col).lower() == "is_deleted" and (v is None):
            v = "false"

        # Convert datetimes/pandas types to strings for Sheets
        if v is None:
            row_out.append(None)
        else:
            try:
                import pandas as _pd
                if isinstance(v, (_pd.Timestamp,)):
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
                            row_out.append(v.isoformat())
                        else:
                            row_out.append(v.strftime("%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        row_out.append(v.isoformat())
                    continue
            except Exception:
                pass
            row_out.append(v)

    try:
        append_res = sheet.values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": [row_out]}
        ).execute()
    except HttpError as e:
        return {"status": "error", "message": f"Failed to append row: {e}"}

    appended_row_number = None
    try:
        updated_range = append_res.get("updates", {}).get("updatedRange") or append_res.get("updatedRange")
        if updated_range and "!" in updated_range:
            rng = updated_range.split("!")[1]
            if ":" in rng:
                last = rng.split(":")[-1]
                m = re.search(r"(\d+)", last)
                if m:
                    appended_row_number = int(m.group(1))
    except Exception:
        appended_row_number = None

    return {"status": "ok", "appended_row_number": appended_row_number, "append_response": append_res}
