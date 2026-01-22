"""
Google Sheets export module for writing enrichment results back to Google Sheets.

This module handles exporting completed prospect data from Supabase to a Google Sheet.
"""

import json
import os
from typing import List, Any, Dict, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def load_google_service_account_credentials() -> Dict[str, Any]:
    """
    Load Google service account credentials from environment variables.
    
    Supports two methods (in order of preference):
    1. GOOGLE_SA_JSON_FILE - file path containing the service account JSON
    2. GOOGLE_SA_JSON - raw JSON string
    
    Returns:
        Dictionary containing service account credentials
        
    Raises:
        ValueError: If neither environment variable is set or credentials cannot be loaded
    """
    # Try GOOGLE_SA_JSON_FILE first (preferred)
    json_file_path = os.getenv("GOOGLE_SA_JSON_FILE")
    if json_file_path:
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    credentials = json.load(f)
                print(f"  [SHEET EXPORT] Using Google service account credentials from FILE: {json_file_path}")
                return credentials
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in GOOGLE_SA_JSON_FILE ({json_file_path}): {str(e)}")
            except Exception as e:
                raise ValueError(f"Error reading GOOGLE_SA_JSON_FILE ({json_file_path}): {str(e)}")
        else:
            raise ValueError(f"GOOGLE_SA_JSON_FILE specified but file does not exist: {json_file_path}")
    
    # Fall back to GOOGLE_SA_JSON
    json_string = os.getenv("GOOGLE_SA_JSON")
    if json_string:
        try:
            credentials = json.loads(json_string)
            print("  [SHEET EXPORT] Using Google service account credentials from ENV (GOOGLE_SA_JSON)")
            return credentials
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GOOGLE_SA_JSON: {str(e)}")
    
    # Neither is set
    raise ValueError("Could not create Google Sheets service. Check GOOGLE_SA_JSON_FILE or GOOGLE_SA_JSON environment variables.")


def get_sheets_service():
    """
    Create and return an authenticated Google Sheets API service client.
    
    Returns:
        Google Sheets API service object, or None if configuration is missing/invalid
    """
    try:
        sa_credentials = load_google_service_account_credentials()
        credentials = service_account.Credentials.from_service_account_info(
            sa_credentials,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        service = build('sheets', 'v4', credentials=credentials)
        return service
    except ValueError as e:
        # ValueError from load_google_service_account_credentials has clear error message
        print(f"  [SHEET EXPORT] {str(e)}")
        return None
    except Exception as e:
        print(f"  [SHEET EXPORT] Error creating Sheets service: {str(e)}")
        return None


def get_sheet_id_for_project(project_id: str, supabase_client: Client) -> Optional[str]:
    """
    Get sheet_id for a given project_id.
    
    For recipe runs, project_id is the Google Sheet ID / logical project key.
    We no longer resolve sheet_id via the prompts table.
    
    Args:
        project_id: Project ID (which is the sheet_id for recipe runs)
        supabase_client: Supabase client instance (unused, kept for API compatibility)
        
    Returns:
        project_id as the sheet_id, or None if project_id is invalid/empty.
    """
    if project_id and project_id.strip():
        return project_id.strip()
    return None


def read_tab_as_rows(service, sheet_id: str, tab_name: str) -> List[Dict[str, Any]]:
    """
    Read all data from a Google Sheets tab and return as list of dictionaries.
    
    The first row is treated as headers. Each subsequent row becomes a dict
    mapping header -> cell value.
    
    Args:
        service: Google Sheets API service object
        sheet_id: Google Sheets spreadsheet ID
        tab_name: Name of the tab to read
        
    Returns:
        List of dictionaries, one per data row (empty list if tab is empty or doesn't exist)
    """
    try:
        sheets = service.spreadsheets()
        range_name = f"{tab_name}!A:ZZ"
        
        result = sheets.values().get(
            spreadsheetId=sheet_id,
            range=range_name
        ).execute()
        
        values = result.get('values', [])
        
        if not values:
            return []
        
        # First row is headers
        headers = [str(h).strip() if h else "" for h in values[0]]
        
        # Build list of dicts from remaining rows
        rows = []
        for row_data in values[1:]:
            row_dict = {}
            for i, header in enumerate(headers):
                if i < len(row_data):
                    row_dict[header] = str(row_data[i]).strip() if row_data[i] else ""
                else:
                    row_dict[header] = ""
            rows.append(row_dict)
        
        return rows
        
    except Exception as e:
        print(f"  [SHEET EXPORT] Error reading tab '{tab_name}' from sheet_id={sheet_id}: {str(e)}")
        return []


def write_rows_to_tab(service, sheet_id: str, tab_name: str, rows: List[Dict[str, Any]]) -> None:
    """
    Write rows to a Google Sheets tab, clearing existing data first.
    
    Args:
        service: Google Sheets API service object
        sheet_id: Google Sheets spreadsheet ID
        tab_name: Name of the tab to write to
        rows: List of dictionaries to write (keys become headers)
        
    Raises:
        Exception: If Sheets API operation fails
    """
    sheets = service.spreadsheets()
    range_name = f"{tab_name}!A:ZZ"
    
    # Clear the tab first
    sheets.values().clear(
        spreadsheetId=sheet_id,
        range=range_name
    ).execute()
    
    # If no rows, we're done (tab is already cleared)
    if not rows:
        return
    
    # Build 2D array: first row is headers, subsequent rows are values
    headers = list(rows[0].keys())
    data_rows = [headers]  # First row is headers
    
    for row in rows:
        row_values = []
        for header in headers:
            value = row.get(header)
            if value is None:
                row_values.append('')
            else:
                row_values.append(str(value))
        data_rows.append(row_values)
    
    # Write data starting at A1
    sheets.values().update(
        spreadsheetId=sheet_id,
        range=f"{tab_name}!A1",
        valueInputOption='RAW',
        body={'values': data_rows}
    ).execute()


def update_master_statuses(service, sheet_id: str, tab_name: str, updates: List[Dict[str, Any]]) -> None:
    """
    Update Status and Cost column values in the Master tab for specified rows.
    
    Args:
        service: Google Sheets API service object
        sheet_id: Google Sheets spreadsheet ID
        tab_name: Name of the tab (typically "Master")
        updates: List of dicts with:
            - 'row_index' (1-based, required)
            - 'status' (optional, string)
            - 'cost_usd' (optional, float) - cost in USD for AI tasks
        
    Raises:
        Exception: If Sheets API operation fails
    """
    if not updates:
        return
    
    sheets = service.spreadsheets()
    
    # Read header row to find Status and Cost columns
    header_range = f"{tab_name}!1:1"
    header_result = sheets.values().get(
        spreadsheetId=sheet_id,
        range=header_range
    ).execute()
    
    header_values = header_result.get('values', [])
    if not header_values or not header_values[0]:
        raise ValueError(f"Could not read header row from {tab_name}")
    
    headers = header_values[0]
    
    # Convert column index to A1 notation (A=0, B=1, etc.)
    # For columns beyond Z, we need to handle AA, AB, etc.
    def col_index_to_letter(col_idx):
        """Convert 0-based column index to A1 notation letter(s)."""
        result = ""
        col_idx += 1  # Convert to 1-based
        while col_idx > 0:
            col_idx -= 1
            result = chr(65 + (col_idx % 26)) + result
            col_idx //= 26
        return result
    
    # Find Status column index (0-based)
    status_col_index = None
    for i, header in enumerate(headers):
        if str(header).strip().lower() == "status":
            status_col_index = i
            break
    
    # If Status column doesn't exist, append it
    if status_col_index is None:
        # Append "Status" to header row
        status_col_index = len(headers)
        status_col_letter = col_index_to_letter(status_col_index)
        sheets.values().update(
            spreadsheetId=sheet_id,
            range=f"{tab_name}!{status_col_letter}1",
            valueInputOption='RAW',
            body={'values': [["Status"]]}
        ).execute()
    
    # Find Cost column index (0-based)
    cost_col_index = None
    for i, header in enumerate(headers):
        if str(header).strip().lower() == "cost":
            cost_col_index = i
            break
    
    # If Cost column doesn't exist, append it
    if cost_col_index is None:
        # Append "Cost" to header row (after Status if Status was just added, or at end)
        cost_col_index = len(headers) if status_col_index < len(headers) else len(headers)
        # If Status was just added, Cost goes right after it
        if status_col_index == len(headers) - 1:
            cost_col_index = len(headers)
        cost_col_letter = col_index_to_letter(cost_col_index)
        sheets.values().update(
            spreadsheetId=sheet_id,
            range=f"{tab_name}!{cost_col_letter}1",
            valueInputOption='RAW',
            body={'values': [["Cost"]]}
        ).execute()
    
    status_col_letter = col_index_to_letter(status_col_index)
    cost_col_letter = col_index_to_letter(cost_col_index)
    
    # Update each row's Status and Cost cells
    for update in updates:
        row_index = update.get("row_index")
        
        # Skip if row_index is missing or invalid (should be 1-based, >= 2 for data rows)
        if row_index is None or not isinstance(row_index, int) or row_index < 1:
            continue
        
        # Update Status if provided
        if "status" in update:
            status = update.get("status", "")
            cell_range = f"{tab_name}!{status_col_letter}{row_index}"
            sheets.values().update(
                spreadsheetId=sheet_id,
                range=cell_range,
                valueInputOption='RAW',
                body={'values': [[str(status)]]}
            ).execute()
        
        # Update Cost if provided
        if "cost_usd" in update:
            cost_usd = update.get("cost_usd")
            if cost_usd is not None:
                # Round to 4 decimal places for display
                cost_value = round(float(cost_usd), 4)
                cell_range = f"{tab_name}!{cost_col_letter}{row_index}"
                sheets.values().update(
                    spreadsheetId=sheet_id,
                    range=cell_range,
                    valueInputOption='RAW',
                    body={'values': [[cost_value]]}
                ).execute()


def export_results_to_google_sheets(project_id: str) -> None:
    """
    Export all prospects for a project from Supabase to Google Sheets.
    
    Steps:
    1. Use project_id as sheet_id (for recipe runs, project_id is the sheet ID)
    2. Load Google service account credentials and build a Sheets API client
    3. Query Supabase: SELECT * FROM prospects WHERE project_id=<id> ORDER BY id ASC
    4. Build a 2D list with first row = column names, following rows = values
    5. Clear the 'output' tab fully using spreadsheets.values.clear()
    6. Write the data starting at A1 using spreadsheets.values.update(..., valueInputOption="RAW")
    
    Args:
        project_id: Project ID to export results for (also serves as sheet_id for recipe runs)
        
    Raises:
        Exception: If Sheets API fails (logged but not re-raised to avoid crashing worker)
    """
    try:
        # Validate required environment variables
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("  [SHEET EXPORT] Supabase credentials not set, skipping export")
            return
        
        # Step 1: Use project_id as sheet_id (no longer query prompts table)
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        print(f"  [SHEET EXPORT] Using project_id as sheet_id for project_id={project_id}...")
        sheet_id = get_sheet_id_for_project(project_id, supabase)
        
        if not sheet_id:
            print(f"  [SHEET EXPORT] No sheet_id found for project_id={project_id}, skipping export")
            return
        
        print(f"  [SHEET EXPORT] Using sheet_id={sheet_id}")
        print("  [SHEET EXPORT] Starting export to Google Sheets...")
        
        # Step 2: Build Google Sheets API client
        service = get_sheets_service()
        if not service:
            print("  [SHEET EXPORT] Could not create Sheets service, skipping export")
            return
        
        sheets = service.spreadsheets()
        
        # Step 3: Query Supabase for all prospects
        print(f"  [SHEET EXPORT] Fetching prospects for project_id={project_id}...")
        
        # Fetch all rows with pagination
        all_prospects = []
        offset = 0
        fetch_batch_size = 1000
        
        while True:
            response = (
                supabase.table('prospects')
                .select('*')
                .eq('project_id', project_id)
                .order('id', desc=False)
                .range(offset, offset + fetch_batch_size - 1)
                .execute()
            )
            
            if not response.data:
                break
            
            all_prospects.extend(response.data)
            
            if len(response.data) < fetch_batch_size:
                break
            
            offset += fetch_batch_size
        
        row_count = len(all_prospects)
        print(f"  [SHEET EXPORT] Fetched {row_count} prospects from Supabase")
        
        # Step 4: Build 2D list (first row = headers, following rows = values)
        if row_count == 0:
            # If no rows, clear the tab and exit cleanly
            print("  [SHEET EXPORT] No rows to export, clearing output tab...")
            sheets.values().clear(
                spreadsheetId=sheet_id,
                range='output!A:ZZ'
            ).execute()
            print(f"  [SHEET EXPORT] Export complete for {project_id}: 0 rows → {sheet_id}/output")
            return
        
        # Get column names from first row
        first_row = all_prospects[0]
        column_names = list(first_row.keys())
        
        # Build 2D list: headers first, then data rows
        data_rows = [column_names]  # First row is headers
        
        for prospect in all_prospects:
            row_values = []
            for col_name in column_names:
                value = prospect.get(col_name)
                # Convert None to empty string, other values to string
                if value is None:
                    row_values.append('')
                else:
                    row_values.append(str(value))
            data_rows.append(row_values)
        
        print(f"  [SHEET EXPORT] Built 2D list: {len(data_rows)} rows (1 header + {row_count} data)")
        
        # Step 5: Clear the 'output' tab fully
        print("  [SHEET EXPORT] Clearing output tab...")
        sheets.values().clear(
            spreadsheetId=sheet_id,
            range='output!A:ZZ'
        ).execute()
        
        # Step 6: Write data starting at A1
        print("  [SHEET EXPORT] Writing data to output tab starting at A1...")
        sheets.values().update(
            spreadsheetId=sheet_id,
            range='output!A1',
            valueInputOption='RAW',
            body={'values': data_rows}
        ).execute()
        
        print(f"  [SHEET EXPORT] Export complete for {project_id}: {row_count} rows → {sheet_id}/output")
        
    except Exception as e:
        # Log error but DO NOT crash the worker
        print(f"  [SHEET EXPORT] Error exporting to Google Sheets: {str(e)}")
        print(f"  [SHEET EXPORT] Export failed, but worker continues normally")

