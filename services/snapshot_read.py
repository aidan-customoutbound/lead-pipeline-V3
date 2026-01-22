"""
Snapshot read module for loading work dictionary from Supabase snapshot tables.

This module handles:
- Loading sheet metadata from run_sheets
- Loading headers from run_sheet_headers
- Paged reading of rows from run_sheet_rows
- Converting snapshot data to work dictionary format
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from supabase import Client


def log(message: str) -> None:
    """Print a log message with [SNAPSHOT_READ] prefix."""
    print(f"[SNAPSHOT_READ] {message}")


def load_sheet_rows_keyset(
    supabase: Client,
    project_id: str,
    run_id: int,
    sheet_name: str,
    last_row_index: Optional[int],
    limit: int
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Query rows using keyset pagination (row_index > last_row_index).
    
    Args:
        supabase: Supabase client instance
        project_id: Project ID
        run_id: Run ID
        sheet_name: Sheet name
        last_row_index: Last row_index from previous page (None for first page)
        limit: Maximum number of rows to return
        
    Returns:
        Tuple of:
        - List of row dictionaries (parsed from row_json)
        - New last_row_index (None if no more rows)
    """
    try:
        query = (
            supabase.table("run_sheet_rows")
            .select("row_index, row_json")
            .eq("project_id", project_id)
            .eq("run_id", run_id)
            .eq("sheet_name", sheet_name)
            .order("row_index", desc=False)
            .limit(limit)
        )
        
        # Apply keyset pagination filter
        if last_row_index is not None:
            query = query.gt("row_index", last_row_index)
        
        response = query.execute()
        
        if not response.data:
            return [], None
        
        # Parse row_json and build list of dicts
        rows = []
        new_last_row_index = None
        
        for row in response.data:
            row_index = row.get("row_index")
            row_json_str = row.get("row_json")
            
            if row_json_str:
                try:
                    row_dict = json.loads(row_json_str)
                    rows.append(row_dict)
                    new_last_row_index = row_index
                except json.JSONDecodeError as e:
                    log(f"Warning: Failed to parse row_json for row_index={row_index}: {str(e)}")
                    # Skip this row but continue
                    continue
        
        # If we got fewer rows than limit, we've reached the end
        if len(rows) < limit:
            new_last_row_index = None
        
        return rows, new_last_row_index
        
    except Exception as e:
        log(f"Error querying rows for sheet '{sheet_name}': {str(e)}")
        raise


def load_work_from_snapshot(
    supabase: Client,
    project_id: str,
    run_id: int,
    page_rows: int = 10000
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load work dictionary from Supabase snapshot tables for a given run.
    
    This function:
    1. Queries run_sheets to get sheet names and row counts
    2. For each sheet:
       - Queries run_sheet_headers to get headers_norm and header_map
       - Pages through run_sheet_rows ordered by row_index
       - Converts each row to dict with normalized header keys
    3. Returns full work dict keyed by sheet name
    
    Args:
        supabase: Supabase client instance
        project_id: Project ID
        run_id: Run ID
        page_rows: Page size for row queries (default 10000)
        
    Returns:
        Dictionary mapping sheet_name -> List[Dict[str, Any]] (rows as dicts)
        
    Raises:
        Exception: If snapshot is missing/incomplete or query errors occur
    """
    log(f"Loading work from snapshot: project_id={project_id}, run_id={run_id}")
    
    # Step 1: Query run_sheets to get sheet names and row counts
    try:
        sheets_response = (
            supabase.table("run_sheets")
            .select("sheet_name, n_rows, status")
            .eq("project_id", project_id)
            .eq("run_id", run_id)
            .order("sheet_name", desc=False)
            .execute()
        )
        
        if not sheets_response.data:
            raise Exception(f"No sheets found in snapshot for project_id={project_id}, run_id={run_id}")
        
        sheets_metadata = sheets_response.data
        log(f"Found {len(sheets_metadata)} sheets in snapshot")
        
    except Exception as e:
        error_msg = f"Failed to query run_sheets: {str(e)}"
        log(f"ERROR: {error_msg}")
        raise Exception(error_msg)
    
    # Step 2: Build work dictionary by loading each sheet
    work: Dict[str, List[Dict[str, Any]]] = {}
    
    for sheet_meta in sheets_metadata:
        sheet_name = sheet_meta.get("sheet_name")
        expected_rows = sheet_meta.get("n_rows", 0)
        status = sheet_meta.get("status")
        
        if not sheet_name:
            log(f"Warning: Skipping sheet with missing sheet_name")
            continue
        
        # Check if sheet ingestion failed
        if status == "failed":
            error_msg = f"Sheet '{sheet_name}' has status='failed' in snapshot"
            log(f"ERROR: {error_msg}")
            raise Exception(error_msg)
        
        log(f"Loading sheet '{sheet_name}': expected {expected_rows} rows")
        
        # Step 2a: Query run_sheet_headers to get headers_norm and header_map
        try:
            headers_response = (
                supabase.table("run_sheet_headers")
                .select("headers_norm, header_map")
                .eq("project_id", project_id)
                .eq("run_id", run_id)
                .eq("sheet_name", sheet_name)
                .single()
                .execute()
            )
            
            if not headers_response.data:
                error_msg = f"No headers found for sheet '{sheet_name}' in snapshot"
                log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
            
            headers_data = headers_response.data
            headers_norm_str = headers_data.get("headers_norm")
            header_map_str = headers_data.get("header_map")
            
            if not headers_norm_str:
                error_msg = f"Missing headers_norm for sheet '{sheet_name}'"
                log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
            
            try:
                headers_norm = json.loads(headers_norm_str)
                header_map = json.loads(header_map_str) if header_map_str else {}
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse headers for sheet '{sheet_name}': {str(e)}"
                log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
            
        except Exception as e:
            if "No headers found" in str(e) or "Missing headers_norm" in str(e) or "Failed to parse" in str(e):
                raise
            error_msg = f"Failed to query headers for sheet '{sheet_name}': {str(e)}"
            log(f"ERROR: {error_msg}")
            raise Exception(error_msg)
        
        # Step 2b: Page through run_sheet_rows
        sheet_rows: List[Dict[str, Any]] = []
        last_row_index: Optional[int] = None
        pages_loaded = 0
        
        while True:
            # Load next page
            page_rows_data, new_last_row_index = load_sheet_rows_keyset(
                supabase,
                project_id,
                run_id,
                sheet_name,
                last_row_index,
                page_rows
            )
            
            if not page_rows_data:
                # No more rows
                break
            
            # row_json should already be keyed by normalized headers from ingestion
            # But we verify and use header_map if needed for translation
            for row_dict in page_rows_data:
                # Verify all keys are in headers_norm (or translate if needed using header_map)
                normalized_row = {}
                
                # Build reverse mapping from raw header to normalized header if needed
                raw_to_norm = {}
                if header_map and "raw" in header_map and "norm" in header_map and "mapping" in header_map:
                    raw_headers = header_map.get("raw", [])
                    norm_headers = header_map.get("norm", [])
                    mapping = header_map.get("mapping", {})
                    for raw_idx, raw_header in enumerate(raw_headers):
                        if raw_idx in mapping:
                            norm_idx = mapping[raw_idx]
                            if norm_idx < len(norm_headers):
                                raw_to_norm[raw_header] = norm_headers[norm_idx]
                
                for key, value in row_dict.items():
                    # Key should already be normalized, but check if it's in headers_norm
                    if key in headers_norm:
                        normalized_row[key] = value
                    elif key in raw_to_norm:
                        # Key is a raw header - translate to normalized
                        norm_key = raw_to_norm[key]
                        normalized_row[norm_key] = value
                    else:
                        # Unexpected key - log warning but include it anyway
                        log(f"Warning: Key '{key}' not found in headers_norm or raw headers for sheet '{sheet_name}', including as-is")
                        normalized_row[key] = value
                
                # Ensure all headers_norm keys are present (fill missing with empty string)
                for header_norm in headers_norm:
                    if header_norm not in normalized_row:
                        normalized_row[header_norm] = ""
                
                sheet_rows.append(normalized_row)
            
            pages_loaded += 1
            last_row_index = new_last_row_index
            
            log(f"  Loaded page {pages_loaded}: {len(page_rows_data)} rows (total: {len(sheet_rows)})")
            
            # If we got fewer rows than page size, we've reached the end
            if new_last_row_index is None:
                break
        
        # Verify row count matches expected
        if len(sheet_rows) != expected_rows:
            log(f"Warning: Sheet '{sheet_name}' has {len(sheet_rows)} rows but expected {expected_rows}")
            # Don't fail - just log warning (snapshot might have been updated)
        
        work[sheet_name] = sheet_rows
        log(f"Completed sheet '{sheet_name}': {len(sheet_rows)} rows in {pages_loaded} pages")
    
    log(f"Loaded work from snapshot: {len(work)} sheets, total rows: {sum(len(rows) for rows in work.values())}")
    
    return work
