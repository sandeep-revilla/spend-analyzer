# io_helpers.py -- data I/O helpers (pure functions, no Streamlit runtime objects passed in)
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta

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


def _normalize_range_for_get(range_name: Optional[str]) -> Optional[str]:
    """Normalize a simple sheet/tab name to a wide A1 range so the Sheets API
    doesn't return a trimmed rect missing newly added rows/columns.

    If range_name is None or already contains an A1 range ('!' or ':'), return as-is.
    Otherwise convert 'SheetName' -> 'SheetName!A:AZ'. Adjust columns if you need wider coverage.
    """
    if not range_name:
        return range_name
    if isinstance(range_name, str) and ('!' not in range_name) and (':' not in range_name):
        # Use A:AZ by default to cover more columns so manual rows are not trimmed.
        return f"{range_name}!A:AZ"
    return range_name


def _sheet_name_from_range(range_name: str) -> str:
    """Return just the sheet/tab name portion of a range (before '!')"""
    if not range_name:
        return ""
    if '!' in range_name:
        return range_name.split('!')[0]
    return range_name


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

    # Build service
    if creds_info is not None:
        service = build_sheets_service_from_info(creds_info)
    else:
        service = build_sheets_service_from_file(creds_file)

    # Candidate ranges to try (best-effort to catch manual appends that sometimes fall outside trimmed rect)
    tried_ranges = []
    norm = _normalize_range_for_get(range_name)
    candidate_ranges = [norm] if norm else []
    # If normalized range doesn't include an explicit row number, also try a large explicit row range
    try:
        sheet_name = _sheet_name_from_range(range_name)
        # large explicit fallback to include many rows (safe but limited so not unbounded)
        if sheet_name:
            large_range = f"{sheet_name}!A1:AZ10000"
            if large_range not in candidate_ranges:
                candidate_ranges.append(large_range)
            # also ensure the simpler A:AZ form is included (if not already)
            a_az = f"{sheet_name}!A:AZ"
            if a_az not in candidate_ranges:
                candidate_ranges.append(a_az)
    except Exception:
        # ignore sheet_name failures; we'll just try the normalized range
        pass

    values = []
    last_err = None
    for r in candidate_ranges:
        if not r:
            continue
        tried_ranges.append(r)
        try:
            sheet = service.spreadsheets()
            res = sheet.values().get(spreadsheetId=spreadsheet_id, range=r).execute()
            values = res.get("values", []) or []
            # If we got values that look usable (non-empty or header present) accept it
            if values and len(values) > 0:
                break
            # If no values, continue to next candidate range
        except HttpError as e:
            last_err = e
            # try next candidate range
            continue

    if not values and last_err is not None:
        # If none worked and there was an HttpError, raise a helpful message
        raise RuntimeError(f"Google Sheets API error reading ranges {tried_ranges}: {last_err}")

    # Convert into DataFrame
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
    # Normalize simple sheet/tab names to wide A1 ranges to ensure API returns newly-added rows.
    history_range = _normalize_range_for_get(history_range)
    append_range = _normalize_range_for_get(append_range)

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
    # Normalize simple sheet/tab names to a wide A1 range for reading/writing.
    range_name = _normalize_range_for_get(range_name)

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


# --- NEW: Security helper to prevent Formula Injection ---
def _sanitize_for_formula_injection(s: str) -> str:
    """
    Prepends a single quote to strings that start with formula-like characters
    ('=', '+', '-', '@') to force Google Sheets to treat them as plain text.
    """
    if s.startswith(('=', '+', '-', '@')):
        return "'" + s
    return s
# --- END NEW ---


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
    # Normalize simple sheet/tab names to a wide A1 range for reading/writing.
    range_name = _normalize_range_for_get(range_name)
    history_range = _normalize_range_for_get(history_range)

    # helpers for number/date detection & excel serial conversion
    def _is_number_like(x):
        if isinstance(x, (int, float)):
            return True
        if isinstance(x, str):
            s = x.strip()
            return bool(re.match(r"^\d+(\.\d+)?$", s))
        return False

    def _to_datetime_if_excel_serial(val_float: float) -> datetime:
        # Excel serial date to python datetime:
        # using epoch 1899-12-30 handles Excel serial numbers cross-platform (common approach)
        excel_epoch = datetime(1899, 12, 30)
        return excel_epoch + timedelta(days=val_float)

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

    # case-insensitive key map for new_row_dict
    key_map = {k.lower(): k for k in new_row_dict.keys()}

    for col in header:
        # find value case-insensitively
        v = None
        if col in new_row_dict:
            v = new_row_dict[col]
        else:
            lookup = key_map.get(col.lower())
            if lookup is not None:
                v = new_row_dict.get(lookup)
            else:
                v = None

        # default for is_deleted
        if str(col).lower() == "is_deleted" and (v is None):
            v = "false"

        # Normalize / convert values
        if v is None:
            row_out.append(None)
            continue

        # pandas Timestamp -> python datetime
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

        # python datetime / date
        try:
            from datetime import datetime as _dt, date as _date
            if isinstance(v, (_dt, _date)):
                try:
                    if isinstance(v, _date) and not isinstance(v, _dt):
                        row_out.append(v.isoformat())
                    else:
                        row_out.append(v.strftime("%Y-%m-%d %H:%M:%S"))
                except Exception:
                    row_out.append(v.isoformat())
                continue
        except Exception:
            pass

        # Detect date-like columns
        col_lower = str(col).lower()
        is_date_like_column = ("date" in col_lower) or ("time" in col_lower) or (col_lower in ("timestamp", "datetime"))

        # Numeric or numeric-string -> possibly Excel serial for date-like columns
        if _is_number_like(v) and is_date_like_column:
            try:
                val_float = float(v)
                # Heuristic: convert any numeric in date-like column using Excel serial epoch
                dt = _to_datetime_if_excel_serial(val_float)
                row_out.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
                continue
            except Exception:
                # fallback to next handling
                pass

        # If numeric and not date-like, preserve numeric
        if isinstance(v, (int, float)):
            row_out.append(v)
            continue

        # --- MODIFIED: Added sanitization for strings ---
        if isinstance(v, str):
            s = v.strip()
            # SECURITY FIX: Sanitize to prevent formula injection
            s_sanitized = _sanitize_for_formula_injection(s)
            row_out.append(s_sanitized)
            continue
        # --- END MODIFICATION ---

        # Fallback: attempt isoformat or str()
        try:
            if hasattr(v, "isoformat"):
                row_out.append(v.isoformat())
            else:
                row_out.append(str(v))
        except Exception:
            row_out.append("")

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
