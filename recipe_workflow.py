"""
Master-sheet recipe runner for Google Sheets.

This module provides a pure in-memory recipe engine that:
1. Parses task definitions from Master sheet rows
2. Executes tasks in order on in-memory copies of URLs and Contacts data
3. Produces final results without any I/O operations

All Supabase and Google Sheets I/O is handled by the caller.
"""

import re
import asyncio
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable
from urllib.parse import urlparse
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables for AI client
load_dotenv()


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
TASK_REMOVE_CHARACTERS = "REMOVE_CHARACTERS"
TASK_CONCATENATE = "CONCATENATE"
TASK_MAP = "MAP"
TASK_ASSIGN_OTHER = "ASSIGN_OTHER"
TASK_AI = "AI"

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


def _parse_remove_characters(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Remove characters - (SheetName, ColumnName, "CharactersToRemove")' pattern.
    
    Returns:
        Dict with 'sheet', 'column', and 'characters_to_remove' keys, or None if parse fails
    """
    # Pattern: Remove characters - (SheetName, ColumnName, "CharactersToRemove")
    # Need to handle quoted string for characters_to_remove
    pattern = r'^Remove characters\s*-\s*\(([^,]+),\s*([^,]+),\s*"([^"]+)"\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    column = match.group(2).strip()
    characters_to_remove = match.group(3).strip()
    
    return {"sheet": sheet, "column": column, "characters_to_remove": characters_to_remove}


def _parse_concatenate(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Concatenate - (SheetName, OutputColumnName, SourceColumn1, SourceColumn2, "Separator")' pattern.
    
    Returns:
        Dict with 'sheet', 'output_column', 'source_column1', 'source_column2', and 'separator' keys, or None if parse fails
    """
    # Pattern: Concatenate - (SheetName, OutputColumnName, SourceColumn1, SourceColumn2, "Separator")
    # Need to handle quoted string for separator
    pattern = r'^Concatenate\s*-\s*\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*"([^"]+)"\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    output_column = match.group(2).strip()
    source_column1 = match.group(3).strip()
    source_column2 = match.group(4).strip()
    separator = match.group(5).strip()
    
    return {
        "sheet": sheet,
        "output_column": output_column,
        "source_column1": source_column1,
        "source_column2": source_column2,
        "separator": separator
    }


def _parse_map(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Map - (TargetSheet, TargetKeyColumn, LookupSheet, LookupKeyColumn, LookupValueColumn, TargetOutputColumn)' pattern.
    
    Returns:
        Dict with 'target_sheet', 'target_key_column', 'lookup_sheet', 'lookup_key_column', 'lookup_value_column', and 'target_output_column' keys, or None if parse fails
    """
    # Pattern: Map - (TargetSheet, TargetKeyColumn, LookupSheet, LookupKeyColumn, LookupValueColumn, TargetOutputColumn)
    pattern = r'^Map\s*-\s*\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    target_sheet = match.group(1).strip()
    target_key_column = match.group(2).strip()
    lookup_sheet = match.group(3).strip()
    lookup_key_column = match.group(4).strip()
    lookup_value_column = match.group(5).strip()
    target_output_column = match.group(6).strip()
    
    return {
        "target_sheet": target_sheet,
        "target_key_column": target_key_column,
        "lookup_sheet": lookup_sheet,
        "lookup_key_column": lookup_key_column,
        "lookup_value_column": lookup_value_column,
        "target_output_column": target_output_column
    }


def _parse_assign_other(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Assign other - <SheetName> | <GroupByColumn> | <Source1>:<Dest1> | <Source2>:<Dest2> | ...' pattern.
    
    Syntax example:
        Assign other - Contacts | WebsiteImport | First Name:Other First | Full Name:Other Full | Focus:Other Focus
    
    Returns:
        Dict with 'sheet_name', 'group_column', and 'mappings' (list of {source, dest} dicts) keys, or None if parse fails
    """
    # Check if task name starts with "Assign other -" (case-insensitive)
    if not task_name.lower().startswith("assign other -"):
        return None
    
    # Remove the prefix and split by pipe
    prefix_len = len("Assign other -")
    rest = task_name[prefix_len:].strip()
    
    if not rest:
        return None
    
    # Split by pipe character
    tokens = [token.strip() for token in rest.split("|")]
    
    # Need at least: sheet_name, group_column, and one mapping pair
    if len(tokens) < 3:
        return None
    
    sheet_name = tokens[0]
    group_column = tokens[1]
    
    # Parse mapping pairs (tokens[2:] onwards)
    mappings = []
    for token in tokens[2:]:
        if not token:
            return None  # Empty token is invalid
        
        # Each token must contain exactly one colon
        if ":" not in token:
            return None
        
        parts = token.split(":", 1)  # Split on first colon only
        if len(parts) != 2:
            return None
        
        source = parts[0].strip()
        dest = parts[1].strip()
        
        # Both sides must be non-empty
        if not source or not dest:
            return None
        
        mappings.append({"source": source, "dest": dest})
    
    # Must have at least one mapping pair
    if not mappings:
        return None
    
    return {
        "sheet_name": sheet_name,
        "group_column": group_column,
        "mappings": mappings
    }


def _parse_ai_task(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'AI - <InputSheet> | <OutputSheet> | <OutputColumn> | <ModelName> | """<PromptText>"""' pattern.
    
    Syntax examples:
        Example 1:
            AI - URLs output | URLs output | GPT_Company_Summary | gpt-mini | """
            Summarize what this company does in one sentence.
            
            Company: {Company}
            Website: {Website}
            Short description: {Short Description}
            
            Return a single plain-English sentence with no fluff.
            """
        
        Example 2:
            AI - Contacts output | Contacts output | GPT_Persona | gpt-4o-mini | """
            Based on this person's role, determine their functional persona.
            
            Full Name: {Full Name}
            Title: {Title}
            Company Summary: {Company Summary}
            
            Return a 2-4 word persona label (e.g., 'Product Leader', 'IT Security Exec').
            """
    
    Returns:
        Dict with 'input_sheet_name', 'output_sheet_name', 'output_column_name', 'model_name', and 'prompt_template' keys,
        or None if parse fails
    """
    # Check if task name starts with "AI -" (case-insensitive)
    if not task_name.lower().startswith("ai -"):
        return None
    
    # Remove the prefix and trim
    prefix_len = len("AI -")
    rest = task_name[prefix_len:].strip()
    
    if not rest:
        return None
    
    # Split by pipe character - but we need to be careful because the prompt may contain pipes
    # The prompt is wrapped in triple quotes, so we can find the last token by looking for triple quotes
    # Strategy: split on |, but the last token should contain the prompt with triple quotes
    
    # Find the position of the first triple quote
    first_triple_quote = rest.find('"""')
    if first_triple_quote == -1:
        return None  # No triple quotes found
    
    # Find the position of the last triple quote (must be after the first one)
    last_triple_quote = rest.rfind('"""')
    if last_triple_quote == -1 or last_triple_quote <= first_triple_quote:
        return None  # Invalid triple quote structure
    
    # Split the part before the prompt on |
    before_prompt = rest[:first_triple_quote].strip()
    prompt_with_quotes = rest[first_triple_quote:last_triple_quote + 3].strip()
    
    # Split before_prompt on |
    tokens_before = [token.strip() for token in before_prompt.split("|")]
    
    # We expect exactly 4 tokens before the prompt: input_sheet, output_sheet, output_column, model_name
    if len(tokens_before) != 4:
        return None
    
    input_sheet_name = tokens_before[0]
    output_sheet_name = tokens_before[1]
    output_column_name = tokens_before[2]
    model_name = tokens_before[3]
    
    # Validate that none are empty
    if not input_sheet_name or not output_sheet_name or not output_column_name or not model_name:
        return None
    
    # Extract prompt template by removing outer triple quotes
    if not prompt_with_quotes.startswith('"""') or not prompt_with_quotes.endswith('"""'):
        return None
    
    prompt_template = prompt_with_quotes[3:-3]  # Remove first and last 3 characters (""")
    
    return {
        "input_sheet_name": input_sheet_name,
        "output_sheet_name": output_sheet_name,
        "output_column_name": output_column_name,
        "model_name": model_name,
        "prompt_template": prompt_template
    }


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
                       TASK_FILTER_EXCLUDE, TASK_COUNT_BY, TASK_SORT, TASK_REMOVE_CHARACTERS, TASK_CONCATENATE):
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
    
    elif task_type == TASK_MAP:
        target_sheet = params.get("target_sheet")
        lookup_sheet = params.get("lookup_sheet")
        
        # Both sheets must be output sheets
        if target_sheet not in OUTPUT_SHEETS:
            return f"Row {row_index}: Target sheet '{target_sheet}' must be an output sheet (URLs output or Contacts output)"
        
        if lookup_sheet not in OUTPUT_SHEETS:
            return f"Row {row_index}: Lookup sheet '{lookup_sheet}' must be an output sheet (URLs output or Contacts output)"
    
    elif task_type == TASK_ASSIGN_OTHER:
        sheet_name = params.get("sheet_name")
        
        # Sheet must be an output sheet
        if sheet_name not in OUTPUT_SHEETS:
            return f"Row {row_index}: Operation on '{sheet_name}' is not allowed. Only output sheets (URLs output, Contacts output) can be used for non-copy operations"
    
    elif task_type == TASK_AI:
        input_sheet_name = params.get("input_sheet_name")
        output_sheet_name = params.get("output_sheet_name")
        output_column_name = params.get("output_column_name")
        model_name = params.get("model_name")
        prompt_template = params.get("prompt_template")
        
        # Validate that all required fields are non-empty
        if not input_sheet_name:
            return f"Row {row_index}: AI task input_sheet_name is required"
        if not output_sheet_name:
            return f"Row {row_index}: AI task output_sheet_name is required"
        if not output_column_name:
            return f"Row {row_index}: AI task output_column_name is required"
        if not model_name:
            return f"Row {row_index}: AI task model_name is required"
        if prompt_template is None:
            return f"Row {row_index}: AI task prompt_template is required"
        
        # Input sheet must exist in inputs or work (we'll check at runtime if it's in work)
        # Output sheet should be an output sheet (but we allow any sheet that exists)
        # For now, we just validate they're non-empty strings
    
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
        elif task_name.lower().startswith("remove characters"):
            parsed = _parse_remove_characters(task_name)
            if parsed:
                task_type = TASK_REMOVE_CHARACTERS
                params = parsed
        elif task_name.lower().startswith("concatenate"):
            parsed = _parse_concatenate(task_name)
            if parsed:
                task_type = TASK_CONCATENATE
                params = parsed
        elif task_name.lower().startswith("map"):
            parsed = _parse_map(task_name)
            if parsed:
                task_type = TASK_MAP
                params = parsed
        elif task_name.lower().startswith("assign other"):
            parsed = _parse_assign_other(task_name)
            if parsed:
                task_type = TASK_ASSIGN_OTHER
                params = parsed
        elif task_name.lower().startswith("ai -"):
            parsed = _parse_ai_task(task_name)
            if parsed:
                task_type = TASK_AI
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


def remove_characters(work: Dict[str, List[Dict[str, Any]]],
                      sheet: str,
                      column: str,
                      characters_to_remove: str) -> None:
    """
    Remove specified characters from a column in-place.
    
    For each row in the sheet, removes every occurrence of any character
    in characters_to_remove from the column value. Writes the cleaned value
    back into the same column.
    
    Examples:
        - Input: "Aidan.?" with characters_to_remove=".?" → Output: "Aidan"
        - Input: "Dr. Smith??" with characters_to_remove=".?" → Output: "Dr Smith"
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        column: Column name to clean (modified in-place)
        characters_to_remove: String containing characters to remove
        
    Raises:
        ValueError: If the column doesn't exist in the sheet
    """
    if sheet not in work:
        work[sheet] = []
    
    rows = work[sheet]
    
    # Validate that the column exists (check if it appears in any row)
    column_exists = any(column in row for row in rows)
    if not column_exists and len(rows) > 0:
        raise ValueError(f"Column '{column}' does not exist in sheet '{sheet}'")
    
    # Build translation table to remove characters
    # str.maketrans creates a translation table that maps each character to None
    if characters_to_remove:
        # Create a translation table: map each char to None (delete)
        trans_table = str.maketrans('', '', characters_to_remove)
    else:
        # No characters to remove, return early
        return
    
    for row in rows:
        value = row.get(column)
        # If the cell is empty/null/undefined, leave it as-is
        if value is None or value == "":
            continue
        
        # Convert to string and remove characters
        value_str = str(value)
        cleaned = value_str.translate(trans_table)
        row[column] = cleaned


def concatenate(work: Dict[str, List[Dict[str, Any]]],
                sheet: str,
                output_column: str,
                source_column1: str,
                source_column2: str,
                separator: str) -> None:
    """
    Concatenate two source columns into an output column.
    
    For each row, combines source_column1 + separator + source_column2
    and writes the result to output_column. If output_column doesn't exist,
    it will be created.
    
    Examples:
        - First Name="Aidan", Last Name="Pits", Separator=" " → "Aidan Pits"
        - First Name="", Company="Stripe", Separator=" at " → " at Stripe"
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        output_column: Column name to write concatenated result to
        source_column1: First source column name
        source_column2: Second source column name
        separator: String to insert between the two values
        
    Raises:
        ValueError: If either source column doesn't exist in the sheet
    """
    if sheet not in work:
        work[sheet] = []
    
    rows = work[sheet]
    
    # Validate that source columns exist (check if they appear in any row)
    if len(rows) > 0:
        source1_exists = any(source_column1 in row for row in rows)
        source2_exists = any(source_column2 in row for row in rows)
        
        if not source1_exists:
            raise ValueError(f"Source column '{source_column1}' does not exist in sheet '{sheet}'")
        if not source2_exists:
            raise ValueError(f"Source column '{source_column2}' does not exist in sheet '{sheet}'")
    
    for row in rows:
        # Fetch values, defaulting to empty string if missing/undefined
        value1 = row.get(source_column1)
        value2 = row.get(source_column2)
        
        # Convert to strings (handles None by converting to "")
        value1_str = str(value1) if value1 is not None else ""
        value2_str = str(value2) if value2 is not None else ""
        
        # Concatenate: value1 + separator + value2
        result = value1_str + separator + value2_str
        
        # Write to output column
        row[output_column] = result


def assign_other_task(work: Dict[str, List[Dict[str, Any]]],
                      sheet_name: str,
                      group_column: str,
                      mappings: List[Dict[str, str]]) -> None:
    """
    Assign "other" colleague information to each row within each group using circular rotation.
    
    This task implements the same logic as the earlier Google Apps Script function ("otherInformation"),
    but generalized and scalable to handle a variable number of column-mapping pairs.
    
    Syntax example:
        Assign other - Contacts | WebsiteImport | First Name:Other First | Full Name:Other Full | Focus:Other Focus
    
    Behavior:
    1. Load all rows for the given sheet from in-memory sheet data.
    2. Group rows by the value in group_column (String(value).trim()).
       - Skip rows with empty grouping key.
    3. For each group:
       - If fewer than 2 valid rows → skip group.
       - Identify "valid rows" as those where ALL source columns exist.
       - Perform circular rotation: For row i → assign data from row (i+1), (i+2), ... until a different row is found.
         (Do NOT assign a row to itself.)
       - If only one unique row has non-empty source fields → skip group.
    4. For each mapping pair <Source>:<Dest>:
       - Read sourceColumn value from colleague row.
       - Write into Dest column of current row.
       - Create the destination column if it doesn't exist yet in the row dict.
       - Leave blank if the assigned colleague has no value.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet_name: Name of sheet in work to process
        group_column: Column name to group rows by
        mappings: List of dicts with 'source' and 'dest' keys for column mappings
    """
    if sheet_name not in work:
        work[sheet_name] = []
    
    rows = work[sheet_name]
    
    # Extract source columns from mappings for validation
    source_columns = [m["source"] for m in mappings]
    
    # Step 1: Group rows by group_column value
    # Convert values to String(value).trim() before grouping
    # Skip rows with empty grouping key
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        group_key_raw = row.get(group_column)
        if group_key_raw is None:
            continue
        
        group_key = str(group_key_raw).strip()
        if not group_key:
            continue  # Skip rows with empty grouping key
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(row)
    
    # Step 2: Process each group
    for group_key, group_rows in groups.items():
        # Step 2a: Filter to valid rows (rows where ALL source columns exist)
        valid_rows = []
        for row in group_rows:
            # Check if all source columns exist in this row
            if all(source_col in row for source_col in source_columns):
                valid_rows.append(row)
        
        # Step 2b: If fewer than 2 valid rows → skip group
        if len(valid_rows) < 2:
            continue
        
        # Step 2c: Check if only one unique row has non-empty source fields
        # We'll check this by comparing the source values
        unique_source_combinations = set()
        for row in valid_rows:
            # Create a tuple of all source values for this row
            source_values = tuple(_safe_str(row.get(src_col)) for src_col in source_columns)
            unique_source_combinations.add(source_values)
        
        # If only one unique combination of source values → skip group
        if len(unique_source_combinations) <= 1:
            continue
        
        # Step 2d: Perform circular rotation
        # For each row i, assign data from row (i+1), wrapping around
        # We need to preserve the original order of valid_rows
        num_valid = len(valid_rows)
        
        for i, current_row in enumerate(valid_rows):
            # Find the next different row (circular)
            # Start from (i+1) and wrap around
            colleague_row = None
            for offset in range(1, num_valid):
                next_idx = (i + offset) % num_valid
                candidate_row = valid_rows[next_idx]
                
                # Check if this candidate has different source values
                current_sources = tuple(_safe_str(current_row.get(src_col)) for src_col in source_columns)
                candidate_sources = tuple(_safe_str(candidate_row.get(src_col)) for src_col in source_columns)
                
                if current_sources != candidate_sources:
                    colleague_row = candidate_row
                    break
            
            # If no different row found, skip this row (shouldn't happen given our check above)
            if colleague_row is None:
                continue
            
            # Step 3: For each mapping pair, assign values
            for mapping in mappings:
                source_col = mapping["source"]
                dest_col = mapping["dest"]
                
                # Read source value from colleague row
                source_value = colleague_row.get(source_col)
                
                # Write to destination column (create if doesn't exist)
                # Leave blank if colleague has no value (None)
                if source_value is None:
                    current_row[dest_col] = ""
                else:
                    current_row[dest_col] = source_value


def map_task(work: Dict[str, List[Dict[str, Any]]],
             target_sheet: str,
             target_key_column: str,
             lookup_sheet: str,
             lookup_key_column: str,
             lookup_value_column: str,
             target_output_column: str) -> None:
    """
    Perform XLOOKUP-style mapping from lookup sheet to target sheet.
    
    Builds a lookup map from lookup_sheet using lookup_key_column as keys
    and lookup_value_column as values. For each row in target_sheet, looks up
    the value in target_key_column and writes the corresponding lookup value
    to target_output_column.
    
    Example:
        URLs output: Website="example.com", B2B/B2C="B2B"
        Contacts output: Website="example.com", B2B/B2C=""
        After Map: Contacts output.B2B/B2C = "B2B"
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        target_sheet: Name of target sheet in work
        target_key_column: Column name in target sheet to use as lookup key
        lookup_sheet: Name of lookup sheet in work
        lookup_key_column: Column name in lookup sheet to use as lookup key
        lookup_value_column: Column name in lookup sheet to use as lookup value
        target_output_column: Column name in target sheet to write lookup result to
        
    Raises:
        ValueError: If any required column doesn't exist in its respective sheet
    """
    if target_sheet not in work:
        work[target_sheet] = []
    if lookup_sheet not in work:
        work[lookup_sheet] = []
    
    lookup_rows = work[lookup_sheet]
    target_rows = work[target_sheet]
    
    # Validate that required columns exist as headers in their respective sheets
    if len(lookup_rows) > 0:
        lookup_key_exists = any(lookup_key_column in row for row in lookup_rows)
        lookup_value_exists = any(lookup_value_column in row for row in lookup_rows)
        
        if not lookup_key_exists:
            raise ValueError(f"Lookup key column '{lookup_key_column}' does not exist in sheet '{lookup_sheet}'")
        if not lookup_value_exists:
            raise ValueError(f"Lookup value column '{lookup_value_column}' does not exist in sheet '{lookup_sheet}'")
    
    if len(target_rows) > 0:
        target_key_exists = any(target_key_column in row for row in target_rows)
        
        if not target_key_exists:
            raise ValueError(f"Target key column '{target_key_column}' does not exist in sheet '{target_sheet}'")
    
    # Build lookup map: lookup_key -> lookup_value
    # Normalize keys by treating them as strings and trimming
    # If multiple rows have the same key, take the FIRST encountered value
    lookup_map: Dict[str, Any] = {}
    for row in lookup_rows:
        key_raw = row.get(lookup_key_column)
        if key_raw is None:
            continue
        
        # Normalize key: String(value).trim() (case-sensitive)
        key = str(key_raw).strip()
        
        # Only add if we haven't seen this key before (first occurrence wins)
        if key and key not in lookup_map:
            value = row.get(lookup_value_column)
            lookup_map[key] = value
    
    # Apply lookup to target rows
    for row in target_rows:
        key_raw = row.get(target_key_column)
        
        # If TargetKeyColumn does not exist on the row or the value is empty/undefined:
        # Leave TargetOutputColumn as-is (do not set an error; just skip)
        if key_raw is None or key_raw == "":
            continue
        
        # Normalize key: String(value).trim()
        key = str(key_raw).strip()
        
        # If key exists in the lookup map, write the value
        if key in lookup_map:
            row[target_output_column] = lookup_map[key]
        # Else: row[target_output_column] remains unchanged (or becomes undefined if not set before)


def _map_model_name(model_name: str) -> str:
    """
    Map user-friendly model names to OpenRouter model identifiers.
    
    Args:
        model_name: User-friendly model name (e.g., "gpt-mini", "gpt-4o-mini", "claude-haiku")
                   or an OpenRouter model identifier (e.g., "openai/gpt-4o-mini")
        
    Returns:
        OpenRouter model identifier string
        
    Raises:
        ValueError: If model_name is not recognized and doesn't look like an OpenRouter identifier
    """
    model_mapping = {
        "gpt-mini": "openai/gpt-4o-mini",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4": "openai/gpt-4-turbo",
        "claude-haiku": "anthropic/claude-3-haiku",
        "claude-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-opus": "anthropic/claude-3-opus",
        "gemini-flash": "google/gemini-2.0-flash-exp",
        "gemini-pro": "google/gemini-pro",
    }
    
    model_name_lower = model_name.lower().strip()
    if model_name_lower in model_mapping:
        return model_mapping[model_name_lower]
    
    # If not found, check if it looks like an OpenRouter model identifier (contains /)
    # Common patterns: "openai/...", "anthropic/...", "google/..."
    if "/" in model_name:
        # Assume it's already an OpenRouter identifier
        return model_name
    
    # Unknown model name - raise error
    raise ValueError(
        f"Unknown model name '{model_name}'. "
        f"Supported models: {', '.join(sorted(model_mapping.keys()))}, "
        f"or use OpenRouter format (e.g., 'openai/gpt-4o-mini')"
    )


async def _call_ai_with_retry(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    max_retries: int = 3,
    backoff_delays: List[float] = None
) -> Optional[str]:
    """
    Call AI model with retry logic and exponential backoff.
    
    Args:
        client: AsyncOpenAI client (configured for OpenRouter)
        prompt: The prompt text to send
        model: Model identifier
        max_retries: Maximum number of retry attempts
        backoff_delays: List of delay seconds for each retry (default: [1, 2])
        
    Returns:
        Response text or None if all attempts fail
    """
    if backoff_delays is None:
        backoff_delays = [1, 2]
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,  # Reasonable default for recipe tasks
                temperature=0.3
            )
            
            if response and response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                await asyncio.sleep(delay)
            else:
                # Last attempt failed
                return None
    
    return None


async def run_ai_task(
    work: Dict[str, List[Dict[str, Any]]],
    task: RecipeTask,
    ai_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> None:
    """
    Execute an AI recipe task.
    
    This function:
    1. Reads input sheet data
    2. Builds prompts for each row by substituting {ColumnName} placeholders
    3. Calls AI API with batching, concurrency limits, and retries
    4. Writes results to output column in output sheet
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        task: RecipeTask with type TASK_AI and params containing:
            - input_sheet_name: Name of input sheet
            - output_sheet_name: Name of output sheet
            - output_column_name: Name of column to write results to
            - model_name: Model name (will be mapped to OpenRouter identifier)
            - prompt_template: Prompt template with {ColumnName} placeholders
        ai_client: AsyncOpenAI client configured for OpenRouter
        semaphore: Semaphore to limit concurrent AI requests
        
    Raises:
        ValueError: If input sheet is not found or other configuration errors
    """
    input_sheet_name = task.params["input_sheet_name"]
    output_sheet_name = task.params["output_sheet_name"]
    output_column_name = task.params["output_column_name"]
    model_name = task.params["model_name"]
    prompt_template = task.params["prompt_template"]
    
    # Resolve sheets
    input_rows = work.get(input_sheet_name)
    if input_rows is None:
        raise ValueError(f"Input sheet '{input_sheet_name}' not found in work dictionary")
    
    output_rows = work.get(output_sheet_name)
    if output_rows is None:
        raise ValueError(f"Output sheet '{output_sheet_name}' not found in work dictionary")
    
    # Ensure row alignment: if input and output are different sheets,
    # we assume they have the same row ordering (row i in input corresponds to row i in output)
    # If they're the same sheet, they refer to the same list
    if input_sheet_name == output_sheet_name:
        output_rows = input_rows
    else:
        # Ensure output_rows has at least as many rows as input_rows
        # If output has fewer rows, extend it with empty dicts
        while len(output_rows) < len(input_rows):
            output_rows.append({})
    
    # Map model name to OpenRouter identifier
    mapped_model = _map_model_name(model_name)
    
    # Determine column headers from first row (if available)
    # We'll use all keys in the row as potential column names
    column_headers = set()
    if len(input_rows) > 0:
        column_headers = set(input_rows[0].keys())
    
    # Build prompts for each row
    async def process_row(row_index: int, row: Dict[str, Any]) -> None:
        """Process a single row: build prompt, call AI, write result."""
        try:
            # Build row values dict (column name -> value)
            row_values = {}
            for col_name in column_headers:
                value = row.get(col_name)
                # Convert to string, handling None/empty
                row_values[col_name] = str(value) if value is not None else ""
            
            # Substitute placeholders in prompt template
            # Pattern: {ColumnName} where ColumnName matches a column header exactly
            # If a column is referenced but doesn't exist, substitute empty string
            prompt = prompt_template
            
            # Find all placeholders in the prompt (pattern: {ColumnName})
            placeholder_pattern = r'\{([^}]+)\}'
            placeholders = re.findall(placeholder_pattern, prompt)
            
            # Replace each placeholder with the corresponding value (or empty string if not found)
            for placeholder_name in placeholders:
                value = row_values.get(placeholder_name, "")
                prompt = prompt.replace(f"{{{placeholder_name}}}", value)
            
            # Call AI with semaphore for concurrency control
            async with semaphore:
                ai_response = await _call_ai_with_retry(ai_client, prompt, mapped_model)
            
            # Write result to output column
            # Ensure the output column exists in the row dict
            # If AI call failed, write empty string (per spec: empty string on failure)
            if ai_response is None:
                output_rows[row_index][output_column_name] = ""
            else:
                output_rows[row_index][output_column_name] = ai_response
        except Exception as e:
            # Per-row error: write empty string and continue processing other rows
            # This ensures one row failure doesn't crash the entire task
            output_rows[row_index][output_column_name] = ""
            # Note: We don't log here since we don't have a logger in this context
            # The caller can handle logging if needed
    
    # Process all rows concurrently (with semaphore limiting concurrency)
    # Batch processing: process in chunks to avoid overwhelming the system
    batch_size = 50
    for batch_start in range(0, len(input_rows), batch_size):
        batch_end = min(batch_start + batch_size, len(input_rows))
        batch_rows = input_rows[batch_start:batch_end]
        
        # Process batch concurrently
        tasks = [
            process_row(batch_start + i, row)
            for i, row in enumerate(batch_rows)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Note: We continue even if some rows fail (exceptions are caught by gather)


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
    
    # Initialize AI client and semaphore (only if we have AI tasks)
    ai_client = None
    ai_semaphore = None
    has_ai_tasks = any(task.type == TASK_AI for task in tasks)
    
    if has_ai_tasks:
        # Initialize OpenRouter client (same pattern as enrich_workflow)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            return {
                "ok": False,
                "errors": ["OPENROUTER_API_KEY environment variable is required for AI tasks"],
                "urls_output": None,
                "contacts_output": None,
                "master_status_updates": None,
            }
        
        ai_client = AsyncOpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "RecipeWorkflow"
            }
        )
        
        # Create semaphore for concurrency control (5 concurrent requests, same as enrichment)
        ai_semaphore = asyncio.Semaphore(5)
    
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
            elif task.type == TASK_REMOVE_CHARACTERS:
                remove_characters(
                    work,
                    task.params["sheet"],
                    task.params["column"],
                    task.params["characters_to_remove"]
                )
            elif task.type == TASK_CONCATENATE:
                concatenate(
                    work,
                    task.params["sheet"],
                    task.params["output_column"],
                    task.params["source_column1"],
                    task.params["source_column2"],
                    task.params["separator"]
                )
            elif task.type == TASK_MAP:
                map_task(
                    work,
                    task.params["target_sheet"],
                    task.params["target_key_column"],
                    task.params["lookup_sheet"],
                    task.params["lookup_key_column"],
                    task.params["lookup_value_column"],
                    task.params["target_output_column"]
                )
            elif task.type == TASK_ASSIGN_OTHER:
                assign_other_task(
                    work,
                    task.params["sheet_name"],
                    task.params["group_column"],
                    task.params["mappings"]
                )
            elif task.type == TASK_AI:
                # AI tasks are async, so we need to run them in an event loop
                # Since run_recipe is sync, we use asyncio.run()
                # Note: asyncio.run() creates a new event loop, so this is safe even if
                # called from a sync context (which is the case in worker.py)
                try:
                    asyncio.run(run_ai_task(work, task, ai_client, ai_semaphore))
                except Exception as e:
                    raise ValueError(f"AI task failed on row {task.row_index}: {str(e)}")
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

