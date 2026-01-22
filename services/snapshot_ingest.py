"""
Snapshot ingestion module for copying Google Sheets data to Supabase snapshot tables.

This module handles:
- Header hardening (normalization, deduplication, blank handling)
- Batch ingestion of sheet data to run_sheets, run_sheet_headers, and run_sheet_rows tables
- Used range detection (scalable, not A:ZZ)
- Supersede-safe batch processing with heartbeat updates
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple
from dotenv import load_dotenv
from supabase import Client

# Load environment variables
load_dotenv()

# Configuration
SNAPSHOT_INGEST_BATCH_ROWS = int(os.getenv("SNAPSHOT_INGEST_BATCH_ROWS", "10000"))


def log(message: str) -> None:
    """Print a log message with [SNAPSHOT] prefix."""
    print(f"[SNAPSHOT] {message}")


def col_index_to_letter(col_idx: int) -> str:
    """
    Convert 0-based column index to A1 notation letter(s).
    
    Examples:
        0 -> A
        1 -> B
        25 -> Z
        26 -> AA
        27 -> AB
    
    Args:
        col_idx: 0-based column index
        
    Returns:
        Column letter(s) in A1 notation
    """
    result = ""
    col_idx += 1  # Convert to 1-based
    while col_idx > 0:
        col_idx -= 1
        result = chr(65 + (col_idx % 26)) + result
        col_idx //= 26
    return result


def harden_headers(raw_headers: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Harden headers by normalizing, handling blanks, and deduplicating.
    
    Rules:
    1. Trim whitespace from all headers
    2. Blank header cell -> "__col_<colIndex>" where colIndex is 1-based
    3. Duplicates -> suffix "__2", "__3", etc.
    
    Args:
        raw_headers: List of raw header strings (may contain blanks, duplicates)
        
    Returns:
        Tuple of:
        - headers_norm: List of normalized header strings (no blanks, no duplicates)
        - header_map: Dict describing the transformation with keys:
            - raw: List of raw headers
            - norm: List of normalized headers
            - mapping: Dict mapping raw_index -> norm_index
            - blank_indices: List of indices where raw header was blank
            - duplicate_map: Dict mapping duplicate header -> list of normalized names
    """
    headers_norm = []
    header_map = {
        "raw": raw_headers.copy(),
        "norm": [],
        "mapping": {},
        "blank_indices": [],
        "duplicate_map": {}
    }
    
    # Track header name counts for deduplication
    header_counts: Dict[str, int] = {}
    
    for i, raw_header in enumerate(raw_headers):
        # Trim whitespace
        trimmed = raw_header.strip() if raw_header else ""
        
        # Handle blank headers
        if not trimmed:
            # Use 1-based column index
            col_index = i + 1
            normalized = f"__col_{col_index}"
            header_map["blank_indices"].append(i)
        else:
            normalized = trimmed
        
        # Handle duplicates
        if normalized in header_counts:
            header_counts[normalized] += 1
            suffix = f"__{header_counts[normalized]}"
            normalized = f"{normalized}{suffix}"
        else:
            header_counts[normalized] = 1
        
        # Track duplicates for header_map
        base_name = normalized.split("__")[0] if "__" in normalized else normalized
        if base_name not in header_map["duplicate_map"]:
            header_map["duplicate_map"][base_name] = []
        header_map["duplicate_map"][base_name].append(normalized)
        
        headers_norm.append(normalized)
        header_map["mapping"][i] = len(headers_norm) - 1
    
    header_map["norm"] = headers_norm
    
    return headers_norm, header_map


def determine_used_range(service, sheet_id: str, sheet_name: str) -> Tuple[int, int]:
    """
    Determine the used range of a sheet (n_cols, n_rows).
    
    Strategy:
    1. Read header row (row 1) to determine n_cols (trim trailing empties)
    2. Read a sentinel column to determine n_rows:
       - Use first non-empty header column if possible
       - Otherwise use column A
    3. Return (n_cols, n_rows) where n_rows >= 1
    
    Args:
        service: Google Sheets API service object
        sheet_id: Google Sheets spreadsheet ID
        sheet_name: Name of the sheet tab
        
    Returns:
        Tuple of (n_cols, n_rows) where both are >= 1
    """
    sheets = service.spreadsheets()
    
    # Step 1: Read header row to determine n_cols
    header_range = f"{sheet_name}!1:1"
    try:
        header_result = sheets.values().get(
            spreadsheetId=sheet_id,
            range=header_range
        ).execute()
        
        header_values = header_result.get('values', [])
        if not header_values or not header_values[0]:
            # Empty sheet - return minimal range
            return (1, 1)
        
        headers = header_values[0]
        # Trim trailing empty headers
        n_cols = len(headers)
        while n_cols > 0 and (n_cols > len(headers) or not headers[n_cols - 1] or not str(headers[n_cols - 1]).strip()):
            n_cols -= 1
        n_cols = max(1, n_cols)  # At least 1 column
        
    except Exception as e:
        log(f"Error reading header row for {sheet_name}: {str(e)}")
        return (1, 1)
    
    # Step 2: Determine sentinel column and read it to find n_rows
    # Choose first non-empty header column, or column A if all are empty
    sentinel_col_idx = 0
    for i, header in enumerate(headers[:n_cols]):
        if header and str(header).strip():
            sentinel_col_idx = i
            break
    
    sentinel_col_letter = col_index_to_letter(sentinel_col_idx)
    sentinel_range = f"{sheet_name}!{sentinel_col_letter}:{sentinel_col_letter}"
    
    try:
        sentinel_result = sheets.values().get(
            spreadsheetId=sheet_id,
            range=sentinel_range
        ).execute()
        
        sentinel_values = sentinel_result.get('values', [])
        # n_rows is the length of returned values (includes header row)
        # We want data rows, so n_rows = len(sentinel_values) (which includes header)
        # But we'll return total rows including header
        n_rows = max(1, len(sentinel_values))
        
    except Exception as e:
        log(f"Error reading sentinel column for {sheet_name}: {str(e)}")
        n_rows = 1
    
    return (n_cols, n_rows)


def read_sheet_batch(service, sheet_id: str, sheet_name: str, start_row: int, end_row: int, n_cols: int) -> List[List[Any]]:
    """
    Read a batch of rows from a sheet.
    
    Args:
        service: Google Sheets API service object
        sheet_id: Google Sheets spreadsheet ID
        sheet_name: Name of the sheet tab
        start_row: 1-based start row (inclusive)
        end_row: 1-based end row (inclusive)
        n_cols: Number of columns to read
        
    Returns:
        List of rows, where each row is a list of cell values
    """
    sheets = service.spreadsheets()
    
    # Build range: A{start_row}:{endCol}{end_row}
    start_col_letter = "A"
    end_col_letter = col_index_to_letter(n_cols - 1)
    range_name = f"{sheet_name}!{start_col_letter}{start_row}:{end_col_letter}{end_row}"
    
    try:
        result = sheets.values().get(
            spreadsheetId=sheet_id,
            range=range_name,
            valueRenderOption='UNFORMATTED_VALUE'  # Get values only, no formulas
        ).execute()
        
        values = result.get('values', [])
        return values
        
    except Exception as e:
        log(f"Error reading batch {start_row}-{end_row} from {sheet_name}: {str(e)}")
        raise


def update_run_heartbeat(supabase: Client, run_id: int) -> None:
    """
    Update runs.heartbeat_at to current timestamp.
    
    Args:
        supabase: Supabase client instance
        run_id: Run ID
    """
    try:
        supabase.table("runs").update({
            "heartbeat_at": datetime.utcnow().isoformat()
        }).eq("id", run_id).execute()
    except Exception as e:
        log(f"Warning: Failed to update heartbeat for run {run_id}: {str(e)}")


def update_run_ingestion_progress(
    supabase: Client,
    run_id: int,
    phase: Optional[str] = None,
    ingested_sheets_total: Optional[int] = None,
    ingested_sheets_done: Optional[int] = None,
    ingested_rows_done: Optional[int] = None
) -> None:
    """
    Update run ingestion progress fields.
    
    Args:
        supabase: Supabase client instance
        run_id: Run ID
        phase: Optional phase to set ('ingesting' or 'processing')
        ingested_sheets_total: Optional total number of sheets
        ingested_sheets_done: Optional number of sheets completed
        ingested_rows_done: Optional number of rows ingested
    """
    try:
        update_data = {}
        if phase is not None:
            update_data["phase"] = phase
        if ingested_sheets_total is not None:
            update_data["ingested_sheets_total"] = ingested_sheets_total
        if ingested_sheets_done is not None:
            update_data["ingested_sheets_done"] = ingested_sheets_done
        if ingested_rows_done is not None:
            update_data["ingested_rows_done"] = ingested_rows_done
        
        if update_data:
            supabase.table("runs").update(update_data).eq("id", run_id).execute()
    except Exception as e:
        log(f"Warning: Failed to update ingestion progress for run {run_id}: {str(e)}")


def ingest_spreadsheet_to_supabase(
    project_id: str,
    run_id: int,
    sheets_service,
    supabase: Client,
    is_run_active_callback: Callable[[], bool],
    heartbeat_callback: Optional[Callable[[], None]] = None
) -> None:
    """
    Ingest all sheet tabs from a Google Spreadsheet into Supabase snapshot tables.
    
    This function:
    1. Gets list of all tabs in the spreadsheet
    2. For each tab:
       - Determines used range (n_cols, n_rows)
       - Reads header row and hardens headers
       - Ingests data in batches
       - Updates progress counters
    3. Updates run phase and timestamps
    4. Handles errors gracefully (marks run_sheets.status='failed' and fails run)
    
    Args:
        project_id: Project ID (also the spreadsheet ID)
        run_id: Run ID
        sheets_service: Google Sheets API service object
        supabase: Supabase client instance
        is_run_active_callback: Callable that returns True if run is still active
        heartbeat_callback: Optional callable to invoke for lock heartbeat (called once per batch)
        
    Raises:
        Exception: If ingestion fails (run will be marked as failed)
    """
    log(f"Starting snapshot ingestion for run {run_id}, project {project_id}")
    
    # Set phase to 'ingesting' and set ingestion_started_at
    try:
        supabase.table("runs").update({
            "phase": "ingesting",
            "ingestion_started_at": datetime.utcnow().isoformat()
        }).eq("id", run_id).execute()
    except Exception as e:
        log(f"Warning: Failed to set ingestion phase: {str(e)}")
    
    # Update heartbeat
    update_run_heartbeat(supabase, run_id)
    
    # Get list of all sheets (tabs) in the spreadsheet
    try:
        spreadsheet_metadata = sheets_service.spreadsheets().get(
            spreadsheetId=project_id,
            includeGridData=False
        ).execute()
        
        sheets_list = spreadsheet_metadata.get('sheets', [])
        tab_titles = [sheet['properties']['title'] for sheet in sheets_list]
        log(f"Found {len(tab_titles)} tabs: {', '.join(tab_titles)}")
        
    except Exception as e:
        error_msg = f"Failed to get spreadsheet metadata: {str(e)}"
        log(f"ERROR: {error_msg}")
        # Mark run as failed
        try:
            supabase.table("runs").update({
                "status": "failed",
                "phase": None,
                "error_message": error_msg[:500],
                "finished_at": datetime.utcnow().isoformat()
            }).eq("id", run_id).eq("status", "running").execute()
        except:
            pass
        raise Exception(error_msg)
    
    # Check if run is still active
    if not is_run_active_callback():
        log(f"Run {run_id} is no longer active, stopping ingestion")
        return
    
    # Update total sheets count
    update_run_ingestion_progress(
        supabase, run_id,
        ingested_sheets_total=len(tab_titles)
    )
    
    total_rows_ingested = 0
    sheets_done = 0
    
    # Process each tab
    for sheet_idx, sheet_name in enumerate(tab_titles):
        log(f"Processing sheet {sheet_idx + 1}/{len(tab_titles)}: '{sheet_name}'")
        
        # Check if run is still active before processing each sheet
        if not is_run_active_callback():
            log(f"Run {run_id} is no longer active, stopping ingestion")
            return
        
        # Update heartbeat
        update_run_heartbeat(supabase, run_id)
        
        try:
            # Determine used range
            n_cols, n_rows = determine_used_range(sheets_service, project_id, sheet_name)
            log(f"  Used range: {n_cols} cols, {n_rows} rows")
            
            # Read header row
            header_batch = read_sheet_batch(sheets_service, project_id, sheet_name, 1, 1, n_cols)
            if not header_batch or not header_batch[0]:
                log(f"  Warning: Empty header row for '{sheet_name}', skipping")
                # Still create run_sheets entry with status='failed'
                try:
                    supabase.table("run_sheets").upsert({
                        "project_id": project_id,
                        "run_id": run_id,
                        "sheet_name": sheet_name,
                        "range_a1": f"A1:A1",
                        "n_rows": 0,
                        "n_cols": 0,
                        "status": "failed",
                        "error": "Empty header row"
                    }, on_conflict="project_id,run_id,sheet_name").execute()
                except Exception as e:
                    log(f"  Warning: Failed to create run_sheets entry: {str(e)}")
                continue
            
            raw_headers = header_batch[0]
            # Pad or trim to n_cols
            while len(raw_headers) < n_cols:
                raw_headers.append("")
            raw_headers = raw_headers[:n_cols]
            
            # Harden headers
            headers_norm, header_map = harden_headers(raw_headers)
            log(f"  Headers normalized: {len(headers_norm)} columns")
            
            # Store run_sheet_headers
            try:
                supabase.table("run_sheet_headers").upsert({
                    "project_id": project_id,
                    "run_id": run_id,
                    "sheet_name": sheet_name,
                    "headers_raw": json.dumps(raw_headers),
                    "headers_norm": json.dumps(headers_norm),
                    "header_map": json.dumps(header_map)
                }, on_conflict="project_id,run_id,sheet_name").execute()
            except Exception as e:
                log(f"  Warning: Failed to store headers: {str(e)}")
            
            # Calculate data rows (excluding header row)
            data_row_count = max(0, n_rows - 1)
            
            # Determine range A1 notation
            end_col_letter = col_index_to_letter(n_cols - 1)
            range_a1 = f"A1:{end_col_letter}{n_rows}"
            
            # Initialize run_sheets entry
            try:
                supabase.table("run_sheets").upsert({
                    "project_id": project_id,
                    "run_id": run_id,
                    "sheet_name": sheet_name,
                    "range_a1": range_a1,
                    "n_rows": data_row_count,
                    "n_cols": n_cols,
                    "status": "processing",
                    "error": None
                }, on_conflict="project_id,run_id,sheet_name").execute()
            except Exception as e:
                log(f"  Warning: Failed to create run_sheets entry: {str(e)}")
            
            # Process data rows in batches
            sheet_rows_ingested = 0
            batch_size = SNAPSHOT_INGEST_BATCH_ROWS
            last_heartbeat_time = time.time()
            
            # Start from row 2 (row 1 is headers)
            for batch_start in range(2, n_rows + 1, batch_size):
                # Check if run is still active before each batch
                if not is_run_active_callback():
                    log(f"Run {run_id} is no longer active, stopping ingestion")
                    # Mark sheet as failed
                    try:
                        supabase.table("run_sheets").update({
                            "status": "failed",
                            "error": "Run superseded during ingestion"
                        }).eq("project_id", project_id).eq("run_id", run_id).eq("sheet_name", sheet_name).execute()
                    except:
                        pass
                    return
                
                # Update heartbeat before each batch (satisfies "at least once per batch")
                # Also check if 15 seconds have passed since last update
                current_time = time.time()
                if current_time - last_heartbeat_time >= 15.0:
                    update_run_heartbeat(supabase, run_id)
                    last_heartbeat_time = current_time
                
                batch_end = min(batch_start + batch_size - 1, n_rows)
                log(f"  Reading batch: rows {batch_start}-{batch_end}")
                
                # Read batch
                batch_data = read_sheet_batch(sheets_service, project_id, sheet_name, batch_start, batch_end, n_cols)
                
                # Convert to row_json format
                rows_to_insert = []
                for row_idx, row_values in enumerate(batch_data):
                    # Pad row to n_cols if needed
                    while len(row_values) < n_cols:
                        row_values.append("")
                    row_values = row_values[:n_cols]
                    
                    # Build row_json dict keyed by headers_norm
                    row_json = {}
                    for col_idx, header_norm in enumerate(headers_norm):
                        if col_idx < len(row_values):
                            # Convert value to string, handle None
                            cell_value = row_values[col_idx]
                            if cell_value is None:
                                row_json[header_norm] = ""
                            else:
                                row_json[header_norm] = str(cell_value)
                        else:
                            row_json[header_norm] = ""
                    
                    # row_index is 1-based data row index (row 2 = index 1, row 3 = index 2, etc.)
                    data_row_index = batch_start - 1 + row_idx
                    
                    rows_to_insert.append({
                        "project_id": project_id,
                        "run_id": run_id,
                        "sheet_name": sheet_name,
                        "row_index": data_row_index,
                        "row_json": json.dumps(row_json)
                    })
                
                # Upsert batch
                if rows_to_insert:
                    try:
                        supabase.table("run_sheet_rows").upsert(
                            rows_to_insert,
                            on_conflict="project_id,run_id,sheet_name,row_index"
                        ).execute()
                        sheet_rows_ingested += len(rows_to_insert)
                        total_rows_ingested += len(rows_to_insert)
                        log(f"  Ingested {len(rows_to_insert)} rows (total for sheet: {sheet_rows_ingested})")
                    except Exception as e:
                        error_msg = f"Failed to upsert batch rows {batch_start}-{batch_end}: {str(e)}"
                        log(f"  ERROR: {error_msg}")
                        # Mark sheet as failed
                        try:
                            supabase.table("run_sheets").update({
                                "status": "failed",
                                "error": error_msg[:500]
                            }).eq("project_id", project_id).eq("run_id", run_id).eq("sheet_name", sheet_name).execute()
                        except:
                            pass
                        # Fail the entire run
                        raise Exception(error_msg)
                
                # Update progress
                update_run_ingestion_progress(
                    supabase, run_id,
                    ingested_rows_done=total_rows_ingested
                )
                
                # Call lock heartbeat callback if provided (once per batch)
                if heartbeat_callback:
                    try:
                        heartbeat_callback()
                    except Exception as e:
                        log(f"  ERROR: Lock heartbeat callback failed: {str(e)}")
                        # If heartbeat fails, abort ingestion
                        raise Exception(f"Lock heartbeat failed during ingestion: {str(e)}")
                
                # Update heartbeat after batch if 15 seconds have passed
                current_time = time.time()
                if current_time - last_heartbeat_time >= 15.0:
                    update_run_heartbeat(supabase, run_id)
                    last_heartbeat_time = current_time
            
            # Mark sheet as completed
            try:
                supabase.table("run_sheets").update({
                    "status": "completed",
                    "error": None
                }).eq("project_id", project_id).eq("run_id", run_id).eq("sheet_name", sheet_name).execute()
            except Exception as e:
                log(f"  Warning: Failed to mark sheet as completed: {str(e)}")
            
            sheets_done += 1
            log(f"  Completed sheet '{sheet_name}': {sheet_rows_ingested} rows")
            
            # Update progress
            update_run_ingestion_progress(
                supabase, run_id,
                ingested_sheets_done=sheets_done
            )
            
        except Exception as e:
            error_msg = f"Failed to ingest sheet '{sheet_name}': {str(e)}"
            log(f"ERROR: {error_msg}")
            
            # Mark sheet as failed
            try:
                supabase.table("run_sheets").update({
                    "status": "failed",
                    "error": error_msg[:500]
                }).eq("project_id", project_id).eq("run_id", run_id).eq("sheet_name", sheet_name).execute()
            except:
                pass
            
            # Fail the entire run
            try:
                supabase.table("runs").update({
                    "status": "failed",
                    "phase": None,
                    "error_message": error_msg[:500],
                    "finished_at": datetime.utcnow().isoformat()
                }).eq("id", run_id).eq("status", "running").execute()
            except:
                pass
            
            raise Exception(error_msg)
    
    # All sheets completed successfully
    # Set phase to 'processing' and set ingestion_finished_at
    try:
        supabase.table("runs").update({
            "phase": "processing",
            "ingestion_finished_at": datetime.utcnow().isoformat()
        }).eq("id", run_id).execute()
    except Exception as e:
        log(f"Warning: Failed to set processing phase: {str(e)}")
    
    log(f"Snapshot ingestion completed: {sheets_done} sheets, {total_rows_ingested} rows")
