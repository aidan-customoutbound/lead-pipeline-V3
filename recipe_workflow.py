"""
Master-sheet recipe runner for Google Sheets.

This module provides a pure in-memory recipe engine that:
1. Parses task definitions from Master sheet rows
2. Executes tasks in order on in-memory copies of URLs and Contacts data
3. Produces final results without any I/O operations

All Supabase and Google Sheets I/O is handled by the caller.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable
from urllib.parse import urlparse


# Task type constants
TASK_COPY_SHEET = "COPY_SHEET"
TASK_DEDUPLICATE = "DEDUPLICATE"
TASK_NORMALIZE_URLS = "NORMALIZE_URLS"
TASK_FILTER_INCLUDE = "FILTER_INCLUDE"
TASK_FILTER_EXCLUDE = "FILTER_EXCLUDE"
TASK_FILTER_MATCH = "FILTER_MATCH"
TASK_FILTER_NOT_MATCH = "FILTER_NOT_MATCH"
TASK_COUNT_BY = "COUNT_BY"
TASK_SORT = "SORT"

# Input sheet names (read-only)
INPUT_SHEETS = {"URLs", "Contacts", "Master"}

# Output sheet names (mutable)
OUTPUT_SHEETS = {"URLs output", "Contacts output"}


@dataclass
class RecipeTask:
    """Represents a parsed recipe task from the Master sheet."""
    row_index: int
    type: str
    params: Dict[str, Any]


def _normalize_url(url: str) -> str:
    """
    Normalize a URL: lower-case, strip protocol, strip www, strip path/query/fragment.
    
    Args:
        url: URL string to normalize
        
    Returns:
        Normalized URL (host only, lowercase, no protocol/www)
    """
    if not url:
        return ""
    
    # Convert to lowercase and strip whitespace
    normalized = url.lower().strip()
    
    if not normalized:
        return ""
    
    # Remove protocol (case insensitive)
    for protocol in ['https://', 'http://']:
        if normalized.startswith(protocol):
            normalized = normalized[len(protocol):]
    
    # Parse URL to extract host
    try:
        # If there's no protocol, add http:// temporarily for parsing
        if '://' not in normalized:
            parsed = urlparse(f"http://{normalized}")
        else:
            parsed = urlparse(normalized)
        
        host = parsed.netloc or parsed.path.split('/')[0]
        
        # Remove www. prefix (case insensitive)
        if host.lower().startswith('www.'):
            host = host[4:]
        
        # Remove trailing slash
        host = host.rstrip('/')
        
        return host
    except Exception:
        # Fallback: simple string manipulation
        # Remove www. prefix
        if normalized.startswith('www.'):
            normalized = normalized[4:]
        
        # Remove path/query/fragment (everything after first /)
        if '/' in normalized:
            normalized = normalized.split('/')[0]
        
        # Remove trailing slash
        normalized = normalized.rstrip('/')
        
        return normalized


def _safe_str(value: Any) -> str:
    """Convert a value to a string, handling None and empty values."""
    if value is None:
        return ""
    return str(value).strip()


def _safe_contains(text: str, search: str) -> bool:
    """Case-insensitive contains check, handling None/empty strings."""
    text_str = _safe_str(text).lower()
    search_str = _safe_str(search).lower()
    return search_str in text_str


def _parse_copy_sheet(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Copy sheet - (SourceSheet) to (TargetSheet)' pattern.
    
    Returns:
        Dict with 'source' and 'target' keys, or None if parse fails
    """
    pattern = r'^Copy sheet\s*-\s*\(([^)]+)\)\s+to\s+\(([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    source = match.group(1).strip()
    target = match.group(2).strip()
    
    return {"source": source, "target": target}


def _parse_deduplicate(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Deduplicate - (SheetName, ColumnName)' pattern.
    
    Returns:
        Dict with 'sheet' and 'column' keys, or None if parse fails
    """
    pattern = r'^Deduplicate\s*-\s*\(([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    column = match.group(2).strip()
    
    return {"sheet": sheet, "column": column}


def _parse_normalize_urls(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Normalize URLs - (SheetName, ColumnName, OutputColumnName)' pattern.
    
    Returns:
        Dict with 'sheet', 'source_column', and 'output_column' keys, or None if parse fails
    """
    pattern = r'^Normalize URLs\s*-\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    source_column = match.group(2).strip()
    output_column = match.group(3).strip()
    
    return {"sheet": sheet, "source_column": source_column, "output_column": output_column}


def _parse_filter_include(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Filter include - (SheetName, ColumnName, "Text to match")' pattern.
    
    Returns:
        Dict with 'sheet', 'column', and 'text' keys, or None if parse fails
    """
    pattern = r'^Filter include\s*-\s*\(([^,]+),\s*([^,]+),\s*"([^"]+)"\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    column = match.group(2).strip()
    text = match.group(3).strip()
    
    return {"sheet": sheet, "column": column, "text": text}


def _parse_filter_exclude(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Filter exclude - (SheetName, ColumnName, "Text to match")' pattern.
    
    Returns:
        Dict with 'sheet', 'column', and 'text' keys, or None if parse fails
    """
    pattern = r'^Filter exclude\s*-\s*\(([^,]+),\s*([^,]+),\s*"([^"]+)"\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    column = match.group(2).strip()
    text = match.group(3).strip()
    
    return {"sheet": sheet, "column": column, "text": text}


def _parse_filter_match(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Filter match - (SourceSheet, SourceColumn, LookupSheet, LookupColumn)' pattern.
    
    Returns:
        Dict with 'source_sheet', 'source_column', 'lookup_sheet', and 'lookup_column' keys, or None if parse fails
    """
    pattern = r'^Filter match\s*-\s*\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    source_sheet = match.group(1).strip()
    source_column = match.group(2).strip()
    lookup_sheet = match.group(3).strip()
    lookup_column = match.group(4).strip()
    
    return {
        "source_sheet": source_sheet,
        "source_column": source_column,
        "lookup_sheet": lookup_sheet,
        "lookup_column": lookup_column
    }


def _parse_filter_not_match(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Filter not match - (SourceSheet, SourceColumn, LookupSheet, LookupColumn)' pattern.
    
    Returns:
        Dict with 'source_sheet', 'source_column', 'lookup_sheet', and 'lookup_column' keys, or None if parse fails
    """
    pattern = r'^Filter not match\s*-\s*\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    source_sheet = match.group(1).strip()
    source_column = match.group(2).strip()
    lookup_sheet = match.group(3).strip()
    lookup_column = match.group(4).strip()
    
    return {
        "source_sheet": source_sheet,
        "source_column": source_column,
        "lookup_sheet": lookup_sheet,
        "lookup_column": lookup_column
    }


def _parse_count_by(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Count by - (SheetName, GroupColumn, OutputColumnName)' pattern.
    
    Returns:
        Dict with 'sheet', 'group_column', and 'output_column' keys, or None if parse fails
    """
    pattern = r'^Count by\s*-\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    group_column = match.group(2).strip()
    output_column = match.group(3).strip()
    
    return {"sheet": sheet, "group_column": group_column, "output_column": output_column}


def _parse_sort(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Sort - (SheetName, ColumnName, Direction)' pattern.
    
    Returns:
        Dict with 'sheet', 'column', and 'direction' keys, or None if parse fails
    """
    pattern = r'^Sort\s*-\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    column = match.group(2).strip()
    direction = match.group(3).strip().upper()
    
    if direction not in ("ASC", "DESC"):
        return None
    
    return {"sheet": sheet, "column": column, "direction": direction}


def _validate_task(task_type: str, params: Dict[str, Any], row_index: int) -> Optional[str]:
    """
    Validate a parsed task according to the rules.
    
    Args:
        task_type: Task type constant
        params: Parsed parameters
        row_index: Row index for error messages
        
    Returns:
        Error message string if validation fails, None if valid
    """
    if task_type == TASK_COPY_SHEET:
        source = params.get("source")
        target = params.get("target")
        
        # Source must be an input sheet
        if source not in INPUT_SHEETS:
            return f"Row {row_index}: Copy sheet source must be an input sheet (URLs, Contacts, or Master), got '{source}'"
        
        # Target must be an output sheet
        if target not in OUTPUT_SHEETS:
            return f"Row {row_index}: Copy sheet target must be an output sheet (URLs output or Contacts output), got '{target}'"
        
        # Cannot copy from input to input
        if target in INPUT_SHEETS:
            return f"Row {row_index}: Copy sheet target cannot be an input sheet, got '{target}'"
    
    elif task_type in (TASK_DEDUPLICATE, TASK_NORMALIZE_URLS, TASK_FILTER_INCLUDE, 
                       TASK_FILTER_EXCLUDE, TASK_COUNT_BY, TASK_SORT):
        sheet = params.get("sheet")
        
        # Sheet must be an output sheet
        if sheet not in OUTPUT_SHEETS:
            return f"Row {row_index}: Operation on '{sheet}' is not allowed. Only output sheets (URLs output, Contacts output) can be used for non-copy operations"
    
    elif task_type in (TASK_FILTER_MATCH, TASK_FILTER_NOT_MATCH):
        source_sheet = params.get("source_sheet")
        lookup_sheet = params.get("lookup_sheet")
        
        # Both sheets must be output sheets
        if source_sheet not in OUTPUT_SHEETS:
            return f"Row {row_index}: Source sheet '{source_sheet}' must be an output sheet (URLs output or Contacts output)"
        
        if lookup_sheet not in OUTPUT_SHEETS:
            return f"Row {row_index}: Lookup sheet '{lookup_sheet}' must be an output sheet (URLs output or Contacts output)"
    
    return None


def parse_master_tasks(master_rows: List[Dict[str, Any]], *, data_row_start: int = 2) -> Tuple[List[RecipeTask], List[str]]:
    """
    Parse Master sheet rows into a list of RecipeTask objects.
    
    Args:
        master_rows: List of row dictionaries from Master sheet
        data_row_start: Starting row index (default 2, since row 1 is header)
        
    Returns:
        Tuple of (tasks, errors):
        - tasks: List of RecipeTask objects
        - errors: List of error message strings (empty if no errors)
    """
    tasks: List[RecipeTask] = []
    errors: List[str] = []
    
    for i, row in enumerate(master_rows):
        row_index = data_row_start + i
        
        # Get Task Name and Status
        task_name_raw = row.get("Task Name", "")
        status_raw = row.get("Status", "")
        
        # Skip empty task names
        if not task_name_raw or not str(task_name_raw).strip():
            continue
        
        # Skip completed tasks
        if str(status_raw).strip().upper() == "COMPLETED":
            continue
        
        task_name = str(task_name_raw).strip()
        
        # Try to parse the task name
        task_type: Optional[str] = None
        params: Optional[Dict[str, Any]] = None
        
        # Try each parser
        if task_name.lower().startswith("copy sheet"):
            parsed = _parse_copy_sheet(task_name)
            if parsed:
                task_type = TASK_COPY_SHEET
                params = parsed
        elif task_name.lower().startswith("deduplicate"):
            parsed = _parse_deduplicate(task_name)
            if parsed:
                task_type = TASK_DEDUPLICATE
                params = parsed
        elif task_name.lower().startswith("normalize urls"):
            parsed = _parse_normalize_urls(task_name)
            if parsed:
                task_type = TASK_NORMALIZE_URLS
                params = parsed
        elif task_name.lower().startswith("filter include"):
            parsed = _parse_filter_include(task_name)
            if parsed:
                task_type = TASK_FILTER_INCLUDE
                params = parsed
        elif task_name.lower().startswith("filter exclude"):
            parsed = _parse_filter_exclude(task_name)
            if parsed:
                task_type = TASK_FILTER_EXCLUDE
                params = parsed
        elif task_name.lower().startswith("filter match"):
            parsed = _parse_filter_match(task_name)
            if parsed:
                task_type = TASK_FILTER_MATCH
                params = parsed
        elif task_name.lower().startswith("filter not match"):
            parsed = _parse_filter_not_match(task_name)
            if parsed:
                task_type = TASK_FILTER_NOT_MATCH
                params = parsed
        elif task_name.lower().startswith("count by"):
            parsed = _parse_count_by(task_name)
            if parsed:
                task_type = TASK_COUNT_BY
                params = parsed
        elif task_name.lower().startswith("sort"):
            parsed = _parse_sort(task_name)
            if parsed:
                task_type = TASK_SORT
                params = parsed
        
        # Check if parsing succeeded
        if task_type is None or params is None:
            errors.append(f"Row {row_index}: Could not parse task name '{task_name}'. Check format matches expected pattern.")
            continue
        
        # Validate the task
        validation_error = _validate_task(task_type, params, row_index)
        if validation_error:
            errors.append(validation_error)
            continue
        
        # Create task
        task = RecipeTask(
            row_index=row_index,
            type=task_type,
            params=params
        )
        tasks.append(task)
    
    return tasks, errors


def copy_sheet(inputs: Dict[str, List[Dict[str, Any]]],
               work: Dict[str, List[Dict[str, Any]]],
               source_sheet: str,
               target_sheet: str) -> None:
    """
    Copy a sheet from inputs to work (creates a deep copy of rows).
    
    Args:
        inputs: Dictionary mapping sheet names to their row lists
        work: Dictionary mapping sheet names to their row lists (mutated)
        source_sheet: Name of source sheet in inputs
        target_sheet: Name of target sheet in work
    """
    if source_sheet not in inputs:
        raise ValueError(f"Source sheet '{source_sheet}' not found in inputs")
    
    # Deep copy the rows
    source_rows = inputs[source_sheet]
    work[target_sheet] = [row.copy() for row in source_rows]


def deduplicate(work: Dict[str, List[Dict[str, Any]]],
                sheet: str,
                column: str) -> None:
    """
    Remove duplicate rows based on a column value (keeps first occurrence).
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        column: Column name to use for deduplication
    """
    if sheet not in work:
        work[sheet] = []
    
    rows = work[sheet]
    seen = set()
    unique_rows = []
    
    for row in rows:
        value = _safe_str(row.get(column))
        if value not in seen:
            seen.add(value)
            unique_rows.append(row)
    
    work[sheet] = unique_rows


def normalize_urls(work: Dict[str, List[Dict[str, Any]]],
                   sheet: str,
                   source_column: str,
                   output_column: str) -> None:
    """
    Normalize URLs from source_column and write to output_column.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        source_column: Column name containing URLs to normalize
        output_column: Column name to write normalized URLs to
    """
    if sheet not in work:
        work[sheet] = []
    
    rows = work[sheet]
    
    for row in rows:
        url = row.get(source_column)
        normalized = _normalize_url(_safe_str(url))
        row[output_column] = normalized


def filter_include(work: Dict[str, List[Dict[str, Any]]],
                  sheet: str,
                  column: str,
                  text: str) -> None:
    """
    Filter rows to keep only those where column contains text (case-insensitive).
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        column: Column name to filter on
        text: Text to search for
    """
    if sheet not in work:
        work[sheet] = []
    
    rows = work[sheet]
    filtered = [row for row in rows if _safe_contains(row.get(column), text)]
    work[sheet] = filtered


def filter_exclude(work: Dict[str, List[Dict[str, Any]]],
                  sheet: str,
                  column: str,
                  text: str) -> None:
    """
    Filter rows to keep only those where column does NOT contain text (case-insensitive).
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        column: Column name to filter on
        text: Text to exclude
    """
    if sheet not in work:
        work[sheet] = []
    
    rows = work[sheet]
    filtered = [row for row in rows if not _safe_contains(row.get(column), text)]
    work[sheet] = filtered


def filter_match(work: Dict[str, List[Dict[str, Any]]],
                source_sheet: str,
                source_column: str,
                lookup_sheet: str,
                lookup_column: str) -> None:
    """
    Filter source_sheet to keep only rows where source_column value exists in lookup_sheet's lookup_column (inner join).
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        source_sheet: Name of source sheet in work
        source_column: Column name in source sheet
        lookup_sheet: Name of lookup sheet in work
        lookup_column: Column name in lookup sheet
    """
    if source_sheet not in work:
        work[source_sheet] = []
    if lookup_sheet not in work:
        work[lookup_sheet] = []
    
    # Build set of lookup values (case-insensitive)
    lookup_rows = work[lookup_sheet]
    lookup_values = set()
    for row in lookup_rows:
        value = _safe_str(row.get(lookup_column)).lower()
        if value:
            lookup_values.add(value)
    
    # Filter source rows
    source_rows = work[source_sheet]
    filtered = []
    for row in source_rows:
        value = _safe_str(row.get(source_column)).lower()
        if value in lookup_values:
            filtered.append(row)
    
    work[source_sheet] = filtered


def filter_not_match(work: Dict[str, List[Dict[str, Any]]],
                    source_sheet: str,
                    source_column: str,
                    lookup_sheet: str,
                    lookup_column: str) -> None:
    """
    Filter source_sheet to keep only rows where source_column value does NOT exist in lookup_sheet's lookup_column (anti-join).
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        source_sheet: Name of source sheet in work
        source_column: Column name in source sheet
        lookup_sheet: Name of lookup sheet in work
        lookup_column: Column name in lookup sheet
    """
    if source_sheet not in work:
        work[source_sheet] = []
    if lookup_sheet not in work:
        work[lookup_sheet] = []
    
    # Build set of lookup values (case-insensitive)
    lookup_rows = work[lookup_sheet]
    lookup_values = set()
    for row in lookup_rows:
        value = _safe_str(row.get(lookup_column)).lower()
        if value:
            lookup_values.add(value)
    
    # Filter source rows
    source_rows = work[source_sheet]
    filtered = []
    for row in source_rows:
        value = _safe_str(row.get(source_column)).lower()
        if value not in lookup_values:
            filtered.append(row)
    
    work[source_sheet] = filtered


def count_by(work: Dict[str, List[Dict[str, Any]]],
            sheet: str,
            group_column: str,
            output_column: str) -> None:
    """
    Count rows by group_column and write count to output_column.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        group_column: Column name to group by
        output_column: Column name to write count to
    """
    if sheet not in work:
        work[sheet] = []
    
    rows = work[sheet]
    
    # Count occurrences of each group value
    counts: Dict[str, int] = {}
    for row in rows:
        value = _safe_str(row.get(group_column)).lower()
        counts[value] = counts.get(value, 0) + 1
    
    # Write counts back to rows
    for row in rows:
        value = _safe_str(row.get(group_column)).lower()
        row[output_column] = counts.get(value, 0)


def sort_rows(work: Dict[str, List[Dict[str, Any]]],
              sheet: str,
              column: str,
              direction: str) -> None:
    """
    Sort rows by column in ascending or descending order.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        column: Column name to sort by
        direction: "ASC" or "DESC" (case-insensitive)
    """
    if sheet not in work:
        work[sheet] = []
    
    rows = work[sheet]
    direction_upper = direction.upper()
    
    def sort_key(row: Dict[str, Any]) -> Any:
        value = row.get(column)
        # Handle None and empty values
        if value is None:
            return ""
        # Try to convert to number if possible
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value).lower()
    
    reverse = (direction_upper == "DESC")
    rows.sort(key=sort_key, reverse=reverse)


def run_recipe(project_id: str,
               run_id: str,
               urls_rows: List[Dict[str, Any]],
               contacts_rows: List[Dict[str, Any]],
               master_rows: List[Dict[str, Any]],
               progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Main entry point for the recipe engine.
    
    This function:
    1. Parses tasks from master_rows
    2. Executes tasks in order on in-memory data
    3. Returns results without performing any I/O
    
    Args:
        project_id: Project ID (for logging only, not used for I/O)
        run_id: Run ID (for logging only, not used for I/O)
        urls_rows: Snapshot of URLs sheet rows
        contacts_rows: Snapshot of Contacts sheet rows
        master_rows: Snapshot of Master sheet rows
        progress_callback: Optional callback function(row_index: int, status: str) called after each task completes
        
    Returns:
        Dictionary with keys:
        - ok: bool (True if successful, False if errors)
        - errors: List[str] (error messages, empty if ok=True)
        - urls_output: List[Dict[str, Any]] or None (final URLs output rows)
        - contacts_output: List[Dict[str, Any]] or None (final Contacts output rows)
        - master_status_updates: List[Dict[str, int]] or None (list of {row_index, status} dicts)
    """
    # Build inputs dictionary
    inputs = {
        "URLs": urls_rows,
        "Contacts": contacts_rows,
        "Master": master_rows,
    }
    
    # Build initial work dictionary (empty)
    work: Dict[str, List[Dict[str, Any]]] = {}
    
    # Parse tasks
    tasks, errors = parse_master_tasks(master_rows)
    
    if errors:
        return {
            "ok": False,
            "errors": errors,
            "urls_output": None,
            "contacts_output": None,
            "master_status_updates": None,
        }
    
    # Execute tasks in order
    master_status_updates = []
    
    try:
        for task in tasks:
            if task.type == TASK_COPY_SHEET:
                copy_sheet(
                    inputs,
                    work,
                    task.params["source"],
                    task.params["target"]
                )
            elif task.type == TASK_DEDUPLICATE:
                deduplicate(
                    work,
                    task.params["sheet"],
                    task.params["column"]
                )
            elif task.type == TASK_NORMALIZE_URLS:
                normalize_urls(
                    work,
                    task.params["sheet"],
                    task.params["source_column"],
                    task.params["output_column"]
                )
            elif task.type == TASK_FILTER_INCLUDE:
                filter_include(
                    work,
                    task.params["sheet"],
                    task.params["column"],
                    task.params["text"]
                )
            elif task.type == TASK_FILTER_EXCLUDE:
                filter_exclude(
                    work,
                    task.params["sheet"],
                    task.params["column"],
                    task.params["text"]
                )
            elif task.type == TASK_FILTER_MATCH:
                filter_match(
                    work,
                    task.params["source_sheet"],
                    task.params["source_column"],
                    task.params["lookup_sheet"],
                    task.params["lookup_column"]
                )
            elif task.type == TASK_FILTER_NOT_MATCH:
                filter_not_match(
                    work,
                    task.params["source_sheet"],
                    task.params["source_column"],
                    task.params["lookup_sheet"],
                    task.params["lookup_column"]
                )
            elif task.type == TASK_COUNT_BY:
                count_by(
                    work,
                    task.params["sheet"],
                    task.params["group_column"],
                    task.params["output_column"]
                )
            elif task.type == TASK_SORT:
                sort_rows(
                    work,
                    task.params["sheet"],
                    task.params["column"],
                    task.params["direction"]
                )
            else:
                raise ValueError(f"Unknown task type: {task.type}")
            
            # Record successful execution
            master_status_updates.append({
                "row_index": task.row_index,
                "status": "completed"
            })
            
            # Call progress callback if provided
            if progress_callback is not None:
                try:
                    progress_callback(task.row_index, "COMPLETED")
                except Exception as callback_error:
                    # Log callback errors but don't fail the recipe
                    # Note: We don't have a logger here, so we'll just continue
                    # The caller can handle logging if needed
                    pass
    
    except Exception as e:
        # Catch runtime errors and return immediately
        error_message = str(e)
        return {
            "ok": False,
            "errors": [f"Runtime error on row {task.row_index}: {error_message}"],
            "urls_output": None,
            "contacts_output": None,
            "master_status_updates": None,
        }
    
    # Extract final results
    urls_output_rows = work.get("URLs output") or []
    contacts_output_rows = work.get("Contacts output") or []
    
    return {
        "ok": True,
        "errors": [],
        "urls_output": urls_output_rows,
        "contacts_output": contacts_output_rows,
        "master_status_updates": master_status_updates,
    }

