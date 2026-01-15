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
from exa_py import Exa

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
TASK_FILTER_BLANK = "FILTER_BLANK"
TASK_COUNT_BY = "COUNT_BY"
TASK_SORT = "SORT"
TASK_REMOVE_CHARACTERS = "REMOVE_CHARACTERS"
TASK_REMOVE_TEXT = "REMOVE_TEXT"
TASK_CONCATENATE = "CONCATENATE"
TASK_MAP = "MAP"
TASK_ASSIGN_OTHER = "ASSIGN_OTHER"
TASK_COPY_BY_KEY = "COPY_BY_KEY"
TASK_INSERT_COLUMN = "INSERT_COLUMN"
TASK_COPY_COLUMNS = "COPY_COLUMNS"
TASK_AI = "AI"
TASK_EXA = "EXA"
TASK_COUNT_MATCHES = "COUNT_MATCHES"

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


def _parse_filter_blank(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Filter blank - (SheetName, ColumnName)' pattern.
    
    Returns:
        Dict with 'sheet' and 'column' keys, or None if parse fails
    """
    pattern = r'^Filter blank\s*-\s*\(([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    column = match.group(2).strip()
    
    return {"sheet": sheet, "column": column}


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


def _parse_count_matches(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Count matches - (SOURCE_SHEET, GROUP_COLUMN, TARGET_SHEET, TARGET_COLUMN, COUNT_COLUMN)' pattern.
    
    Returns:
        Dict with 'source_sheet', 'group_column', 'target_sheet', 'target_column', and 'count_column' keys,
        or None if parse fails
    """
    # Match the pattern: "Count matches - (param1, param2, param3, param4, param5)"
    pattern = r'^Count matches\s*-\s*\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    # Extract and trim all 5 parameters
    params = [match.group(i).strip() for i in range(1, 6)]
    
    # Validate that all parameters are non-empty
    if not all(params):
        return None
    
    source_sheet, group_column, target_sheet, target_column, count_column = params
    
    return {
        "source_sheet": source_sheet,
        "group_column": group_column,
        "target_sheet": target_sheet,
        "target_column": target_column,
        "count_column": count_column
    }


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


def _parse_remove_text(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Remove text - (SheetName, ColumnName, Phrase1 | Phrase2 | Phrase3 | ...)' pattern.
    
    Syntax examples:
        Remove text - (Contacts output, Company Name, LLC | Inc. | Corporation | Corp.)
        Remove text - (Contacts output, Company Name, "LLC" | "Inc." | "Corporation")
        Remove text - (Contacts output, Title, Sr. | Jr. | III)
    
    Returns:
        Dict with 'sheet', 'column', and 'phrases' (list[str]) keys, or None if parse fails
    """
    # Pattern: Remove text - (SheetName, ColumnName, Phrase1 | Phrase2 | Phrase3 | ...)
    pattern = r'^Remove text\s*-\s*\(([^,]+),\s*([^,]+),\s*(.+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    column = match.group(2).strip()
    phrases_raw = match.group(3).strip()
    
    # Split on | and filter out empty parts
    parts = [p.strip() for p in phrases_raw.split("|")]
    parts = [p for p in parts if p]  # Filter out empty parts
    
    if not parts:
        return None
    
    # Normalize quotes: strip surrounding single or double quotes if present
    normalized_phrases = []
    for p in parts:
        phrase = p.strip()
        # Check for double quotes
        if phrase.startswith('"') and phrase.endswith('"') and len(phrase) >= 2:
            phrase = phrase[1:-1].strip()
        # Check for single quotes
        elif phrase.startswith("'") and phrase.endswith("'") and len(phrase) >= 2:
            phrase = phrase[1:-1].strip()
        
        if phrase:  # Only add non-empty phrases after normalization
            normalized_phrases.append(phrase)
    
    if not normalized_phrases:
        return None
    
    return {
        "sheet": sheet,
        "column": column,
        "phrases": normalized_phrases,
    }


def _parse_concatenate(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Concatenate - (SheetName, OutputColumn, InputColumn1 | InputColumn2 | InputColumn3 | ...)' pattern.
    
    Returns:
        Dict with 'sheet', 'output_column', and 'input_columns' keys, or None if parse fails
    """
    # Pattern: Concatenate - (SheetName, OutputColumn, InputColumn1 | InputColumn2 | InputColumn3 | ...)
    pattern = r'^Concatenate\s*-\s*\(([^,]+),\s*([^,]+),\s*(.+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    sheet = match.group(1).strip()
    output_column = match.group(2).strip()
    inputs_raw = match.group(3).strip()
    
    # Split on | and filter out empty parts
    parts = [p.strip() for p in inputs_raw.split("|")]
    parts = [p for p in parts if p]  # Filter out empty parts
    
    if not parts:
        return None
    
    return {
        "sheet": sheet,
        "output_column": output_column,
        "input_columns": parts,
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
    Parse 'Assign other - (SheetName, GroupColumn, SourceCol1:DestCol1 | SourceCol2:DestCol2 | ...)' pattern.
    
    Syntax examples:
        Assign other - (Contacts output, Website, First Name:Other First | Last Name:Other Last)
        Assign other - (Contacts output, Website, First Name:Other First)
    
    Returns:
        Dict with 'sheet', 'group_column', and 'mappings' (list of (source_col, dest_col) tuples) keys, or None if parse fails
    """
    # Pattern: Assign other - (SheetName, GroupColumn, SourceCol1:DestCol1 | SourceCol2:DestCol2 | ...)
    # Use regex to capture content between parentheses
    pattern = r'^Assign other\s*-\s*\((.+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    if not match:
        return None
    
    content = match.group(1).strip()
    
    # Split on commas at the top level to extract SheetName, GroupColumn, mappings_string
    # We need to be careful about commas inside column names, so we'll split on comma and take first 3 parts
    parts = [p.strip() for p in content.split(',', 2)]
    
    if len(parts) < 3:
        return None
    
    sheet_name = parts[0].strip()
    group_column = parts[1].strip()
    mappings_string = parts[2].strip()
    
    # Validate that sheet and group_column are non-empty
    if not sheet_name or not group_column:
        return None
    
    # Parse the mappings string (pipe-separated list of SourceCol:DestCol pairs)
    segments = [s.strip() for s in mappings_string.split("|") if s.strip()]
    
    if not segments:
        return None  # Must have at least one mapping
    
    mappings = []
    for segment in segments:
        # Each segment must be of the form SourceCol:DestCol
        if ":" not in segment:
            return None  # Malformed mapping
        
        # Split on first colon only
        colon_pos = segment.find(":")
        if colon_pos == -1 or colon_pos == 0 or colon_pos == len(segment) - 1:
            return None  # Malformed mapping
        
        source_col = segment[:colon_pos].strip()
        dest_col = segment[colon_pos + 1:].strip()
        
        # Both source and dest column names must be non-empty
        if not source_col or not dest_col:
            return None
        
        mappings.append((source_col, dest_col))
    
    # Must have at least one mapping
    if not mappings:
        return None
    
    return {
        "sheet": sheet_name,
        "group_column": group_column,
        "mappings": mappings
    }


def _parse_copy_by_key(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Copy by key - (SourceSheet, TargetSheet, SourceKeyColumn, TargetKeyColumn)' pattern.
    
    Syntax examples:
        Copy by key - (Acct, Contacts output, Website, Website)
        Copy by key - (URLs output, Contacts output, Website, Website)
    
    Returns:
        Dict with 'source_sheet', 'target_sheet', 'source_key_column', and 'target_key_column' keys, or None if parse fails
    """
    # Use regex to match: Copy by key - (SourceSheet, TargetSheet, SourceKeyColumn, TargetKeyColumn)
    # Tolerant of whitespace around commas and parentheses
    pattern = r'^Copy by key\s*-\s*\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    
    if not match:
        return None
    
    source_sheet = match.group(1).strip()
    target_sheet = match.group(2).strip()
    source_key_column = match.group(3).strip()
    target_key_column = match.group(4).strip()
    
    # Validate that all extracted strings are non-empty
    if not source_sheet or not target_sheet or not source_key_column or not target_key_column:
        return None
    
    return {
        "source_sheet": source_sheet,
        "target_sheet": target_sheet,
        "source_key_column": source_key_column,
        "target_key_column": target_key_column,
    }


def _parse_insert_column(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Insert column - (SheetName, ColumnName)' pattern.
    
    Syntax examples:
        Insert column - (Contacts output, Situational)
        insert column-(URLs output, Score)
        INSERT COLUMN - ( DNC URL , Flag )
    
    Returns:
        Dict with 'sheet' and 'column' keys, or None if parse fails
    """
    # Use regex to match: Insert column - (Sheet, Column)
    # Case-insensitive, flexible whitespace
    pattern = r'^Insert column\s*-\s*\(([^,]+),\s*([^)]+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    
    if not match:
        return None
    
    sheet = match.group(1).strip()
    column = match.group(2).strip()
    
    # Validate that both are non-empty
    if not sheet or not column:
        return None
    
    return {
        "sheet": sheet,
        "column": column
    }


def _parse_copy_columns(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Copy columns - (SourceSheet, TargetSheet, SourceCol1:TargetCol1 | SourceCol2:TargetCol2 | ...)' pattern.
    
    Syntax examples:
        Copy columns - (Contacts, Contacts output, Email:Email)
        Copy columns - (Contacts, Contacts output, First Name:First Name | Last Name:Last Name | Email:Email)
        Copy columns - (Acct, URLs output, Account Website:Website | Account Name:Company Name)
    
    Returns:
        Dict with 'source_sheet', 'target_sheet', and 'mappings' keys (mappings is a list of (src_col, tgt_col) tuples),
        or None if parse fails
    """
    # Check if task name starts with "Copy columns -" (case-insensitive)
    if not task_name.lower().startswith("copy columns -"):
        return None
    
    # Remove the prefix and trim
    prefix_len = len("Copy columns -")
    rest = task_name[prefix_len:].strip()
    
    if not rest:
        return None
    
    # Pattern: (SourceSheet, TargetSheet, SourceCol1:TargetCol1 | SourceCol2:TargetCol2 | ...)
    pattern = r'^\(([^,]+),\s*([^,]+),\s*(.+)\)$'
    match = re.match(pattern, rest, re.IGNORECASE)
    if not match:
        return None
    
    source_sheet = match.group(1).strip()
    target_sheet = match.group(2).strip()
    mappings_str = match.group(3).strip()
    
    # Validate that source and target sheets are non-empty
    if not source_sheet or not target_sheet:
        return None
    
    # Parse the mappings string (pipe-separated list of SourceCol:TargetCol pairs)
    mappings = []
    mapping_parts = [part.strip() for part in mappings_str.split("|")]
    
    for mapping_part in mapping_parts:
        if not mapping_part:
            continue  # Skip empty parts
        
        # Each mapping must contain a colon
        if ":" not in mapping_part:
            return None  # Malformed mapping
        
        # Split by colon (only split on first colon in case column names contain colons)
        colon_pos = mapping_part.find(":")
        if colon_pos == -1 or colon_pos == 0 or colon_pos == len(mapping_part) - 1:
            return None  # Malformed mapping
        
        src_col = mapping_part[:colon_pos].strip()
        tgt_col = mapping_part[colon_pos + 1:].strip()
        
        # Both source and target column names must be non-empty
        if not src_col or not tgt_col:
            return None
        
        mappings.append((src_col, tgt_col))
    
    # Must have at least one mapping
    if not mappings:
        return None
    
    return {
        "source_sheet": source_sheet,
        "target_sheet": target_sheet,
        "mappings": mappings
    }


def _parse_exa_condition(condition_segment: str) -> Optional[Dict[str, str]]:
    """
    Parse a WHEN condition segment: 'WHEN {ColumnName} is not empty'
    
    Args:
        condition_segment: The condition segment string (e.g., 'WHEN {Short Description} is not empty')
        
    Returns:
        Dict with 'column_name' and 'type' keys, or None if parse fails
    """
    condition_segment = condition_segment.strip()
    
    # Must start with WHEN (case-insensitive)
    if not condition_segment.lower().startswith("when"):
        return None
    
    # Remove WHEN prefix and trim
    after_when = condition_segment[4:].strip()
    if not after_when:
        return None
    
    # Expect placeholder {ColumnName}
    if not after_when.startswith("{") or "}" not in after_when:
        return None
    
    # Extract column name from placeholder
    placeholder_end = after_when.find("}")
    if placeholder_end == -1:
        return None
    
    column_name = after_when[1:placeholder_end].strip()
    if not column_name:
        return None
    
    # After placeholder, expect space and "is not empty" (case-insensitive)
    after_placeholder = after_when[placeholder_end + 1:].strip()
    after_placeholder_lower = after_placeholder.lower()
    
    # Check if it matches "is not empty" (case-insensitive, allowing extra spaces)
    if "is not empty" not in after_placeholder_lower:
        return None
    
    # Normalize: remove "is not empty" and check if only whitespace remains
    # This ensures the tail is exactly "is not empty" (with optional surrounding spaces)
    normalized_tail = after_placeholder_lower.replace("is not empty", "").strip()
    if normalized_tail:
        # There's extra text after "is not empty" - invalid
        return None
    
    return {
        "column_name": column_name,
        "type": "is_not_empty"
    }


def _parse_ai_condition(condition_segment: str) -> Optional[Dict[str, str]]:
    """
    Parse a WHEN condition segment: 'WHEN ColumnName contains "Text"'
    
    Args:
        condition_segment: The condition segment string (e.g., 'WHEN Title contains "manager"')
        
    Returns:
        Dict with 'column_name' and 'substring' keys, or None if parse fails
    """
    condition_segment = condition_segment.strip()
    
    # Must start with WHEN (case-insensitive)
    if not condition_segment.lower().startswith("when"):
        return None
    
    # Remove WHEN prefix and trim
    after_when = condition_segment[4:].strip()
    if not after_when:
        return None
    
    # Use regex to match: ColumnName contains "Text"
    # Pattern: column name (alphanumeric, spaces, underscores, hyphens), then "contains", then quoted string
    pattern = r'^([A-Za-z0-9 _-]+)\s+contains\s+"([^"]+)"$'
    match = re.match(pattern, after_when, re.IGNORECASE)
    
    if not match:
        return None
    
    column_name = match.group(1).strip()
    substring = match.group(2).strip()
    
    if not column_name or not substring:
        return None
    
    return {
        "column_name": column_name,
        "substring": substring
    }


def _parse_ai_task(task_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse AI task pattern with optional WHEN condition.
    
    Supports two formats:
    1. Unconditional: 'AI - (InputSheet, OutputSheet, OutputColumn, Model) """<PromptText>"""'
    2. Conditional: 'AI - (InputSheet, OutputSheet, OutputColumn, Model) WHEN ColumnName contains "Text" """<PromptText>"""'
    
    Syntax examples:
        Example 1 (unconditional):
            AI - (Contacts output, Contacts output, GPT_Title, openai/gpt-4o-mini) \"\"\"
            Summarize the job title in one sentence.
            Company: {Company}
            \"\"\"
        
        Example 2 (conditional):
            AI - (Contacts output, Contacts output, GPT_Situational, openai/gpt-4o-mini) WHEN Title contains "manager" \"\"\"
            Summarize the job title in one sentence.
            Company: {Company}
            \"\"\"
    
    Returns:
        Dict with 'input_sheet', 'output_sheet', 'output_column', 'model', 'prompt_template',
        'condition_column' (None or str), 'condition_substring' (None or str),
        or None if parse fails
    """
    # Check if task text starts with "AI -" (case-insensitive)
    if not task_text.lower().startswith("ai -"):
        return None
    
    # Find the first line up to the starting """
    # The DSL line may be on the first line, and the prompt may start on the same or next line
    first_triple_quote = task_text.find('"""')
    if first_triple_quote == -1:
        return None  # No triple quotes found
    
    # Extract the DSL line (everything before the first """)
    dsl_line = task_text[:first_triple_quote].strip()
    
    # Remove "AI -" prefix
    prefix_len = len("AI -")
    if len(dsl_line) < prefix_len:
        return None
    
    rest = dsl_line[prefix_len:].strip()
    if not rest:
        return None
    
    # Extract the parenthesized section after "AI -"
    # Pattern: (content) with optional WHEN clause before triple quotes
    # First, find the closing parenthesis
    if not rest.startswith("("):
        return None
    
    paren_end = rest.find(")")
    if paren_end == -1:
        return None
    
    inner = rest[1:paren_end].strip()
    after_parens = rest[paren_end + 1:].strip()
    
    # Split inner content on commas
    parts = [p.strip() for p in inner.split(",")]
    
    # Expect exactly 4 parts
    if len(parts) != 4:
        return None
    
    input_sheet = parts[0]
    output_sheet = parts[1]
    output_column = parts[2]
    model = parts[3]
    
    # Validate that none are empty
    if not input_sheet or not output_sheet or not output_column or not model:
        return None
    
    # Parse optional WHEN clause
    condition_column = None
    condition_substring = None
    
    if after_parens:
        # Must start with WHEN if there's any text after parentheses
        if not after_parens.lower().startswith("when"):
            return None  # Unexpected text after parentheses
        
        condition = _parse_ai_condition(after_parens)
        if condition is not None:
            condition_column = condition["column_name"]
            condition_substring = condition["substring"]
        else:
            # Invalid condition syntax
            return None
    
    # Extract prompt template
    # Find the first """ and the matching closing """
    last_triple_quote = task_text.rfind('"""')
    if last_triple_quote == -1 or last_triple_quote <= first_triple_quote:
        return None  # Invalid triple quote structure
    
    prompt_with_quotes = task_text[first_triple_quote:last_triple_quote + 3].strip()
    
    if not prompt_with_quotes.startswith('"""') or not prompt_with_quotes.endswith('"""'):
        return None
    
    prompt_template = prompt_with_quotes[3:-3]  # Remove first and last 3 characters (""")
    
    result = {
        "input_sheet": input_sheet,
        "output_sheet": output_sheet,
        "output_column": output_column,
        "model": model,
        "condition_column": condition_column,
        "condition_substring": condition_substring,
        "prompt_template": prompt_template,
    }
    
    return result


def _parse_exa_task(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse Exa task pattern with optional reference column.
    
    Supports two formats (case-insensitive, whitespace-flexible):
    1. Unconditional: 'Exa - (SheetName, WebsiteColumn, OutputColumn)'
    2. Conditional: 'Exa - (SheetName, WebsiteColumn, OutputColumn, ReferenceColumn)'
    
    Syntax examples:
        Example 1 (unconditional):
            Exa - (URLs output, Website, Exa Summary)
        
        Example 2 (conditional):
            Exa - (URLs output, Website, Exa Summary, Needs Exa)
    
    Returns:
        Dict with 'sheet', 'website_column', 'output_column', and optionally 'reference_column',
        or None if parse fails
    """
    # Match pattern: Exa - (content) with case-insensitive matching
    pattern = r'^Exa\s*-\s*\((.+)\)$'
    match = re.match(pattern, task_name, re.IGNORECASE)
    
    if not match:
        return None
    
    # Extract content inside parentheses
    inner = match.group(1).strip()
    if not inner:
        return None
    
    # Split by commas and strip each part
    parts = [p.strip() for p in inner.split(",")]
    
    # Must have either 3 or 4 parts
    if len(parts) not in (3, 4):
        return None
    
    sheet = parts[0]
    website_column = parts[1]
    output_column = parts[2]
    reference_column = parts[3] if len(parts) == 4 else None
    
    # Validate that required fields are non-empty
    if not sheet or not website_column or not output_column:
        return None
    
    # Validate reference_column if provided
    if reference_column is not None and not reference_column:
        return None
    
    result = {
        "sheet": sheet,
        "website_column": website_column,
        "output_column": output_column,
        "reference_column": reference_column
    }
    
    return result


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
        
        # Validate that all required fields are non-empty
        if not source:
            return f"Row {row_index}: Copy sheet source is required"
        if not target:
            return f"Row {row_index}: Copy sheet target is required"
    
    elif task_type in (TASK_DEDUPLICATE, TASK_NORMALIZE_URLS, TASK_FILTER_INCLUDE, 
                       TASK_FILTER_EXCLUDE, TASK_FILTER_BLANK, TASK_COUNT_BY, TASK_SORT, TASK_REMOVE_CHARACTERS):
        sheet = params.get("sheet")
        
        # Validate that all required fields are non-empty
        if not sheet:
            return f"Row {row_index}: Sheet name is required"
    
    elif task_type == TASK_CONCATENATE:
        sheet = params.get("sheet")
        output_column = params.get("output_column")
        input_columns = params.get("input_columns")
        
        if not sheet:
            return f"Row {row_index}: Concatenate task sheet is required"
        if not output_column:
            return f"Row {row_index}: Concatenate task output_column is required"
        if not input_columns or not isinstance(input_columns, list) or len(input_columns) == 0:
            return f"Row {row_index}: Concatenate task must have at least one input column"
        for col in input_columns:
            if not col or not isinstance(col, str) or not col.strip():
                return f"Row {row_index}: Concatenate task input columns must be non-empty strings"
    
    elif task_type == TASK_REMOVE_TEXT:
        sheet = params.get("sheet")
        column = params.get("column")
        phrases = params.get("phrases")
        
        # Validate that all required fields are non-empty
        if not sheet or not isinstance(sheet, str) or not sheet.strip():
            return f"Row {row_index}: Remove text task sheet is required"
        if not column or not isinstance(column, str) or not column.strip():
            return f"Row {row_index}: Remove text task column is required"
        if not phrases or not isinstance(phrases, list) or len(phrases) == 0:
            return f"Row {row_index}: Remove text task must have at least one phrase"
        for phrase in phrases:
            if not phrase or not isinstance(phrase, str) or not phrase.strip():
                return f"Row {row_index}: Remove text task phrases must be non-empty strings"
    
    elif task_type in (TASK_FILTER_MATCH, TASK_FILTER_NOT_MATCH):
        source_sheet = params.get("source_sheet")
        source_column = params.get("source_column")
        lookup_sheet = params.get("lookup_sheet")
        lookup_column = params.get("lookup_column")
        
        # Validate that all required fields are non-empty
        if not source_sheet or not isinstance(source_sheet, str) or not source_sheet.strip():
            return f"Row {row_index}: Filter match task source_sheet is required"
        if not source_column or not isinstance(source_column, str) or not source_column.strip():
            return f"Row {row_index}: Filter match task source_column is required"
        if not lookup_sheet or not isinstance(lookup_sheet, str) or not lookup_sheet.strip():
            return f"Row {row_index}: Filter match task lookup_sheet is required"
        if not lookup_column or not isinstance(lookup_column, str) or not lookup_column.strip():
            return f"Row {row_index}: Filter match task lookup_column is required"
    
    elif task_type == TASK_COUNT_MATCHES:
        source_sheet = params.get("source_sheet")
        group_column = params.get("group_column")
        target_sheet = params.get("target_sheet")
        target_column = params.get("target_column")
        count_column = params.get("count_column")
        
        # Validate that all required fields are non-empty
        if not source_sheet:
            return f"Row {row_index}: Count matches task source_sheet is required"
        if not group_column:
            return f"Row {row_index}: Count matches task group_column is required"
        if not target_sheet:
            return f"Row {row_index}: Count matches task target_sheet is required"
        if not target_column:
            return f"Row {row_index}: Count matches task target_column is required"
        if not count_column:
            return f"Row {row_index}: Count matches task count_column is required"
    
    elif task_type == TASK_MAP:
        target_sheet = params.get("target_sheet")
        lookup_sheet = params.get("lookup_sheet")
        
        # Validate that all required fields are non-empty
        if not target_sheet:
            return f"Row {row_index}: Target sheet is required"
        if not lookup_sheet:
            return f"Row {row_index}: Lookup sheet is required"
    
    elif task_type == TASK_ASSIGN_OTHER:
        sheet = params.get("sheet")
        group_column = params.get("group_column")
        mappings = params.get("mappings")
        
        # Validate that all required fields are non-empty
        if not sheet:
            return f"Assign other: sheet name is required"
        if not group_column:
            return f"Assign other: group column is required"
        if not mappings or not isinstance(mappings, list) or len(mappings) == 0:
            return f"Assign other: at least one Source:Dest mapping is required"
        
        # Validate each mapping
        for i, mapping in enumerate(mappings):
            if not isinstance(mapping, (tuple, list)) or len(mapping) != 2:
                return f"Assign other: mapping {i+1} must be a (source, dest) tuple"
            source_col = mapping[0]
            dest_col = mapping[1]
            if not source_col or not dest_col:
                mapping_str = f"{source_col or '?'}:{dest_col or '?'}"
                return f"Assign other: mapping '{mapping_str}' is missing source or dest column"
    
    elif task_type == TASK_COPY_BY_KEY:
        source_sheet = params.get("source_sheet")
        target_sheet = params.get("target_sheet")
        source_key_column = params.get("source_key_column")
        target_key_column = params.get("target_key_column")
        
        # Validate that all required fields are non-empty
        if not source_sheet:
            return f"Row {row_index}: Copy by key task source_sheet is required"
        if not target_sheet:
            return f"Row {row_index}: Copy by key task target_sheet is required"
        if not source_key_column:
            return f"Row {row_index}: Copy by key task source_key_column is required"
        if not target_key_column:
            return f"Row {row_index}: Copy by key task target_key_column is required"
    
    elif task_type == TASK_INSERT_COLUMN:
        sheet = params.get("sheet")
        column = params.get("column")
        
        # Validate that all required fields are non-empty
        if not sheet:
            return f"Row {row_index}: Insert column task sheet is required"
        if not column:
            return f"Row {row_index}: Insert column task column is required"
    
    elif task_type == TASK_COPY_COLUMNS:
        source_sheet = params.get("source_sheet")
        target_sheet = params.get("target_sheet")
        mappings = params.get("mappings")
        
        # Validate that all required fields are non-empty
        if not source_sheet:
            return f"Row {row_index}: Copy columns task source_sheet is required"
        if not target_sheet:
            return f"Row {row_index}: Copy columns task target_sheet is required"
        if not mappings or not isinstance(mappings, list) or len(mappings) == 0:
            return f"Row {row_index}: Copy columns task must have at least one column mapping"
    
    elif task_type == TASK_AI:
        input_sheet = params.get("input_sheet")
        output_sheet = params.get("output_sheet")
        output_column = params.get("output_column")
        model = params.get("model")
        prompt_template = params.get("prompt_template")
        condition_column = params.get("condition_column")
        condition_substring = params.get("condition_substring")
        
        # Validate that all required fields are non-empty
        if not input_sheet:
            return f"Row {row_index}: AI task input_sheet is required"
        if not output_sheet:
            return f"Row {row_index}: AI task output_sheet is required"
        if not output_column:
            return f"Row {row_index}: AI task output_column is required"
        if not model:
            return f"Row {row_index}: AI task model is required"
        if not prompt_template:
            return f"Row {row_index}: AI task prompt_template is required"
        
        # Validate condition: either both None (unconditional) or both non-empty (conditional)
        if (condition_column is None) != (condition_substring is None):
            return f"Row {row_index}: AI task condition_column and condition_substring must both be present or both be absent"
        
        if condition_column is not None and not condition_column:
            return f"Row {row_index}: AI task condition_column must be non-empty when condition is present"
        
        if condition_substring is not None and not condition_substring:
            return f"Row {row_index}: AI task condition_substring must be non-empty when condition is present"
    
    elif task_type == TASK_EXA:
        sheet = params.get("sheet")
        website_column = params.get("website_column")
        output_column = params.get("output_column")
        reference_column = params.get("reference_column")
        
        # Validate that all required fields are non-empty
        if not sheet:
            return f"Row {row_index}: Exa task sheet is required"
        if not website_column:
            return f"Row {row_index}: Exa task website_column is required"
        if not output_column:
            return f"Row {row_index}: Exa task output_column is required"
        
        # Validate reference_column if present (must be non-empty string)
        if reference_column is not None and not reference_column:
            return f"Row {row_index}: Exa task reference_column must be non-empty if provided"
    
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
        elif task_name.lower().startswith("filter blank"):
            parsed = _parse_filter_blank(task_name)
            if parsed:
                task_type = TASK_FILTER_BLANK
                params = parsed
        elif task_name.lower().startswith("count by"):
            parsed = _parse_count_by(task_name)
            if parsed:
                task_type = TASK_COUNT_BY
                params = parsed
        elif task_name.lower().startswith("count matches"):
            parsed = _parse_count_matches(task_name)
            if parsed:
                task_type = TASK_COUNT_MATCHES
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
        elif task_name.lower().startswith("remove text"):
            parsed = _parse_remove_text(task_name)
            if parsed:
                task_type = TASK_REMOVE_TEXT
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
        elif task_name.lower().startswith("copy by key"):
            parsed = _parse_copy_by_key(task_name)
            if parsed:
                task_type = TASK_COPY_BY_KEY
                params = parsed
        elif task_name.lower().startswith("insert column"):
            parsed = _parse_insert_column(task_name)
            if parsed:
                task_type = TASK_INSERT_COLUMN
                params = parsed
        elif task_name.lower().startswith("copy columns"):
            parsed = _parse_copy_columns(task_name)
            if parsed:
                task_type = TASK_COPY_COLUMNS
                params = parsed
        elif task_name.lower().startswith("ai -"):
            parsed = _parse_ai_task(task_name)
            if parsed:
                task_type = TASK_AI
                params = parsed
        elif task_name.lower().startswith("exa"):
            parsed = _parse_exa_task(task_name)
            if parsed:
                task_type = TASK_EXA
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


def _extract_referenced_sheets(tasks: List[RecipeTask]) -> set:
    """
    Extract all sheet names referenced by tasks.
    
    Args:
        tasks: List of RecipeTask objects
        
    Returns:
        Set of sheet names referenced by any task
    """
    referenced_sheets = set()
    
    for task in tasks:
        params = task.params
        
        if task.type == TASK_COPY_SHEET:
            # Only source is required to exist; target can be created
            referenced_sheets.add(params.get("source"))
        elif task.type == TASK_DEDUPLICATE:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_NORMALIZE_URLS:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_FILTER_INCLUDE:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_FILTER_EXCLUDE:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_FILTER_BLANK:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_FILTER_MATCH:
            referenced_sheets.add(params.get("source_sheet"))
            referenced_sheets.add(params.get("lookup_sheet"))
        elif task.type == TASK_FILTER_NOT_MATCH:
            referenced_sheets.add(params.get("source_sheet"))
            referenced_sheets.add(params.get("lookup_sheet"))
        elif task.type == TASK_COUNT_BY:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_COUNT_MATCHES:
            referenced_sheets.add(params.get("source_sheet"))
            referenced_sheets.add(params.get("target_sheet"))
        elif task.type == TASK_SORT:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_REMOVE_CHARACTERS:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_REMOVE_TEXT:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_CONCATENATE:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_MAP:
            referenced_sheets.add(params.get("target_sheet"))
            referenced_sheets.add(params.get("source_sheet"))
        elif task.type == TASK_ASSIGN_OTHER:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_COPY_BY_KEY:
            referenced_sheets.add(params.get("source_sheet"))
            referenced_sheets.add(params.get("target_sheet"))
        elif task.type == TASK_INSERT_COLUMN:
            referenced_sheets.add(params.get("sheet"))
        elif task.type == TASK_COPY_COLUMNS:
            referenced_sheets.add(params.get("source_sheet"))
            referenced_sheets.add(params.get("target_sheet"))
        elif task.type == TASK_AI:
            referenced_sheets.add(params.get("input_sheet"))
            referenced_sheets.add(params.get("output_sheet"))
        elif task.type == TASK_EXA:
            referenced_sheets.add(params.get("sheet"))
    
    # Remove None values
    referenced_sheets.discard(None)
    return referenced_sheets


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
    print(f"[RECIPE][COPY_SHEET] Source sheet name: {source_sheet}")
    print(f"[RECIPE][COPY_SHEET] Target sheet name: {target_sheet}")
    
    source_exists_in_work = source_sheet in work
    print(f"[RECIPE][COPY_SHEET] Source exists in work: {source_exists_in_work}")
    
    if not source_exists_in_work:
        print(f"[RECIPE][COPY_SHEET] WARNING: Source sheet '{source_sheet}' does not exist in work")
        work[target_sheet] = []
        print(f"[RECIPE][COPY_SHEET] Final count in work[{target_sheet}]: 0")
        return
    
    source_rows = work[source_sheet]
    source_row_count = len(source_rows)
    print(f"[RECIPE][COPY_SHEET] Rows in source: {source_row_count}")
    
    # Deep copy the rows
    work[target_sheet] = [row.copy() for row in source_rows]
    copied_count = len(work[target_sheet])
    print(f"[RECIPE][COPY_SHEET] Rows copied: {copied_count}")
    print(f"[RECIPE][COPY_SHEET] Final count in work[{target_sheet}]: {copied_count}")


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
    rows_before = len(work.get(sheet, []))
    print(f"[RECIPE][NORMALIZE_URLS] sheet='{sheet}', input_column='{source_column}', output_column='{output_column}', rows_before={rows_before}")
    
    if sheet not in work:
        print(f"[RECIPE][NORMALIZE_URLS] WARNING: sheet='{sheet}' not found in work, creating empty sheet")
        work[sheet] = []
    
    rows = work[sheet]
    
    if not rows:
        print(f"[RECIPE][NORMALIZE_URLS] sheet='{sheet}' has no rows, returning without modification")
        return
    
    # Check if source column exists in at least one row
    available_keys = set()
    for row in rows:
        available_keys.update(row.keys())
    
    if source_column not in available_keys:
        print(f"[RECIPE][NORMALIZE_URLS] WARNING: input_column='{source_column}' not found in sheet='{sheet}', available_keys={sorted(available_keys)}")
    
    rows_processed = 0
    for row in rows:
        url = row.get(source_column)
        normalized = _normalize_url(_safe_str(url))
        row[output_column] = normalized
        rows_processed += 1
    
    print(f"[RECIPE][NORMALIZE_URLS] sheet='{sheet}', rows_processed={rows_processed}, rows_after={len(rows)}")


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


def filter_blank(work: Dict[str, List[Dict[str, Any]]],
                 sheet: str,
                 column: str) -> None:
    """
    Filter rows to keep only those where column is NOT blank.
    A "blank" value is defined as None, empty string "", or whitespace-only string.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        column: Column name to filter on
    """
    if sheet not in work:
        print(f"[RECIPE][FILTER_BLANK] sheet='{sheet}' not found in work, returning without modification")
        return
    
    rows = work[sheet]
    
    # Return early if no rows exist
    if not rows:
        print(f"[RECIPE][FILTER_BLANK] sheet='{sheet}' has no rows, returning without modification")
        return
    
    # Normalize the requested column name
    normalized_column = column.strip().lower()
    
    # Build mapping from normalized header names to actual keys
    first_row = rows[0]
    header_map = {}
    for key in first_row.keys():
        normalized_key = key.strip().lower()
        header_map[normalized_key] = key
    
    # Get available keys for logging
    available_keys = list(first_row.keys())
    rows_before = len(rows)
    
    # Log before filtering
    print(f"[RECIPE][FILTER_BLANK] sheet='{sheet}', column='{column}', normalized='{normalized_column}', rows_before={rows_before}, keys={available_keys}")
    
    # Check if column exists
    if normalized_column not in header_map:
        print(f"[RECIPE][FILTER_BLANK] column='{column}' not found in sheet='{sheet}', available_keys={available_keys}")
        return
    
    # Get the actual column key
    actual_column = header_map[normalized_column]
    
    filtered = []
    for row in rows:
        value = row.get(actual_column)
        # Check if value is blank: None, empty string, or whitespace-only
        if value is None:
            continue  # Skip blank rows
        value_str = str(value).strip()
        if value_str:  # Non-empty after strip, keep the row
            filtered.append(row)
    
    rows_after = len(filtered)
    work[sheet] = filtered
    
    # Log after filtering
    print(f"[RECIPE][FILTER_BLANK] sheet='{sheet}', actual_key='{actual_column}', rows_after={rows_after}")


def filter_match(work: Dict[str, List[Dict[str, Any]]],
                source_sheet: str,
                source_column: str,
                lookup_sheet: str,
                lookup_column: str,
                errors: List[str]) -> None:
    """
    Filter source_sheet to keep only rows where source_column value exists in lookup_sheet's lookup_column (inner join).
    
    Keep ONLY rows in SourceSheet where SourceColumn's value is present in LookupSheet.LookupColumn.
    All other rows in SourceSheet are removed.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        source_sheet: Name of source sheet in work
        source_column: Column name in source sheet
        lookup_sheet: Name of lookup sheet in work
        lookup_column: Column name in lookup sheet
        errors: List to append error messages to
    """
    # Check sheet existence
    if source_sheet not in work:
        error_msg = f"Filter match task: source_sheet='{source_sheet}' not found; available sheets={list(work.keys())}"
        print(f"[RECIPE][FILTER_MATCH][ERROR] {error_msg}")
        errors.append(error_msg)
        return
    
    if lookup_sheet not in work:
        error_msg = f"Filter match task: lookup_sheet='{lookup_sheet}' not found; available sheets={list(work.keys())}"
        print(f"[RECIPE][FILTER_MATCH][ERROR] {error_msg}")
        errors.append(error_msg)
        return
    
    # Extract rows
    source_rows = work[source_sheet]
    lookup_rows = work[lookup_sheet]
    
    # Check if source_column exists in source sheet
    if source_rows:
        available_keys = set()
        for row in source_rows:
            available_keys.update(row.keys())
        if source_column not in available_keys:
            error_msg = f"Filter match task: source_column='{source_column}' not found in source_sheet='{source_sheet}'; available columns={sorted(available_keys)}"
            print(f"[RECIPE][FILTER_MATCH][ERROR] {error_msg}")
            errors.append(error_msg)
            return
    
    # Build lookup set from LookupSheet.LookupColumn
    lookup_values = set()
    for row in lookup_rows:
        raw = row.get(lookup_column, "")
        if raw is None:
            raw = ""
        val = str(raw).strip().lower()
        if val:
            lookup_values.add(val)
    
    # Log start
    print(f"[RECIPE][FILTER_MATCH] source_sheet='{source_sheet}', source_column='{source_column}', lookup_sheet='{lookup_sheet}', lookup_column='{lookup_column}', lookup_distinct_values={len(lookup_values)}, rows_source_before={len(source_rows)}")
    
    # If lookup_values is empty, log warning but continue (result will be empty)
    if not lookup_values:
        print(f"[RECIPE][FILTER_MATCH] WARNING: lookup_sheet='{lookup_sheet}', lookup_column='{lookup_column}' has no non-empty values; all rows will be filtered out")
    
    # Filter source rows
    filtered = []
    for row in source_rows:
        raw = row.get(source_column, "")
        if raw is None:
            raw = ""
        key = str(raw).strip().lower()
        if key in lookup_values:
            filtered.append(row)
    
    # Replace work[source_sheet] with filtered list
    work[source_sheet] = filtered
    
    # Log completion
    print(f"[RECIPE][FILTER_MATCH] rows_after={len(filtered)}")


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


def count_matches(work: Dict[str, List[Dict[str, Any]]],
                  source_sheet: str,
                  group_column: str,
                  target_sheet: str,
                  target_column: str,
                  count_column: str) -> None:
    """
    Count matches from source_sheet and write counts to target_sheet.
    
    Builds a frequency dictionary from source_sheet rows grouped by group_column,
    then writes the count for each matching value in target_sheet to count_column.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        source_sheet: Name of source sheet in work (to count from)
        group_column: Column name in source sheet to group by
        target_sheet: Name of target sheet in work (to write counts to)
        target_column: Column name in target sheet to match against
        count_column: Column name in target sheet to write count to
    """
    # Check if source sheet exists
    if source_sheet not in work:
        print(f"[RECIPE][COUNT_MATCHES] Source sheet missing")
        # Note: We don't have access to errors list here, so we just log and return
        # The caller should handle this validation before calling
        return
    
    # Check if target sheet exists
    if target_sheet not in work:
        print(f"[RECIPE][COUNT_MATCHES] Target sheet missing")
        return
    
    source_rows = work[source_sheet]
    target_rows = work[target_sheet]
    
    # Log start
    print(f"[RECIPE][COUNT_MATCHES] sheet_source={source_sheet}, sheet_target={target_sheet}, rows_source={len(source_rows)}, rows_target={len(target_rows)}")
    
    # Step A: Build frequency dictionary from source_sheet rows
    count_dict: Dict[str, int] = {}
    for row in source_rows:
        # Get value from group_column, treat missing as ""
        value = row.get(group_column, "")
        if value is None:
            value = ""
        value = str(value).strip()
        count_dict[value] = count_dict.get(value, 0) + 1
    
    # Log count of distinct keys
    distinct_keys = len(count_dict)
    print(f"[RECIPE][COUNT_MATCHES] distinct_keys={distinct_keys}")
    
    # Step B: For every row in target_sheet, write count
    for row in target_rows:
        # Get key from target_column, treat missing as ""
        key = row.get(target_column, "")
        if key is None:
            key = ""
        key = str(key).strip()
        
        # Get count from dictionary (default to 0 if not found)
        count = count_dict.get(key, 0)
        
        # Write count to count_column (create if doesn't exist, overwrite if exists)
        row[count_column] = count
    
    # Log end
    print(f"[RECIPE][COUNT_MATCHES] completed")


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


def remove_text(
        work: Dict[str, List[Dict[str, Any]]],
        sheet: str,
        column: str,
        phrases: List[str],
        errors: List[str],
    ) -> None:
    """
    Remove specific phrases (substrings) from a column in-place, case-insensitively.
    
    For each row in the sheet, removes all case-insensitive occurrences of each phrase
    from the column value, then writes the cleaned value back into the same column.
    After removal, normalizes whitespace (collapses multiple spaces, strips leading/trailing).
    
    Examples:
        - Input: "Acme LLC" with phrases=["LLC"] → Output: "Acme"
        - Input: "BigCo Inc." with phrases=["Inc."] → Output: "BigCo"
        - Input: "ACME llc" with phrases=["llc"] → Output: "ACME"
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        column: Column name to clean (modified in-place)
        phrases: List of phrase strings to remove (case-insensitive)
        errors: List to append error messages to
        
    Note:
        If sheet or column doesn't exist, logs error and appends to errors list.
        Does not modify work if sheet/column is missing.
    """
    # Check sheet
    if sheet not in work:
        error_msg = f"Remove text: sheet '{sheet}' not found"
        print(f"[RECIPE][REMOVE_TEXT][ERROR] sheet='{sheet}' not found; available sheets={list(work.keys())}")
        errors.append(error_msg)
        return
    
    # Retrieve rows
    rows = work[sheet]
    
    if not rows:
        print(f"[RECIPE][REMOVE_TEXT] sheet='{sheet}' has no rows; nothing to clean")
        return
    
    # Check column presence
    column_exists = any(column in row for row in rows)
    if not column_exists:
        error_msg = f"Remove text: column '{column}' not found in sheet '{sheet}'"
        print(f"[RECIPE][REMOVE_TEXT][ERROR] column='{column}' not found in any rows of sheet='{sheet}'")
        errors.append(error_msg)
        return
    
    # Log start
    print(f"[RECIPE][REMOVE_TEXT] sheet='{sheet}', column='{column}', phrases={phrases}, row_count={len(rows)}")
    
    # Precompile regex patterns for case-insensitive phrase removal
    phrase_patterns = []
    for phrase in phrases:
        if phrase:  # Skip empty phrases (shouldn't happen after validation, but be safe)
            # Escape special regex characters in the phrase
            escaped_phrase = re.escape(phrase)
            # Create a pattern that matches the phrase case-insensitively
            pattern = re.compile(escaped_phrase, re.IGNORECASE)
            phrase_patterns.append(pattern)
    
    # Process each row
    for row in rows:
        raw_value = row.get(column, "")
        
        # If value is None or missing, treat as empty string
        if raw_value is None:
            text = ""
        else:
            text = str(raw_value)
        
        # For each phrase in the phrase list
        for pattern in phrase_patterns:
            # Remove all occurrences of that phrase from the text, case-insensitively
            text = pattern.sub('', text)
        
        # After all phrases are removed:
        # Normalize whitespace: replace sequences of whitespace with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading and trailing whitespace
        text = text.strip()
        
        # If the resulting string is empty, store ""
        if not text:
            text = ""
        
        # Write the cleaned string back into the same ColumnName for that row
        row[column] = text
    
    # Assign back and log
    work[sheet] = rows
    print(f"[RECIPE][REMOVE_TEXT] completed sheet='{sheet}', column='{column}', rows_updated={len(rows)}")


def concatenate(
        work: Dict[str, List[Dict[str, Any]]],
        sheet: str,
        output_column: str,
        input_columns: List[str],
        errors: List[str],
    ) -> None:
    """
    Concatenate multiple input columns into an output column.
    
    For each row:
    - Read values from each InputColumn in the given order.
    - Treat missing or None values as "".
    - Strip whitespace from each input value.
    - Ignore completely empty values when building the output.
    - Join the non-empty pieces with a single space " ".
    - If all input columns are empty, the result is "".
    - Write the result string to OutputColumn in that row.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        output_column: Column name to write concatenated result to
        input_columns: List of input column names to concatenate
        errors: List to append error messages to
    """
    if sheet not in work:
        error_msg = f"Concatenate: sheet '{sheet}' not found"
        print(f"[RECIPE][CONCATENATE][ERROR] sheet='{sheet}' not found; available sheets={list(work.keys())}")
        errors.append(error_msg)
        return
    
    rows = work[sheet]
    
    if not rows:
        print(f"[RECIPE][CONCATENATE] sheet='{sheet}' has no rows; nothing to concatenate")
        return
    
    print(f"[RECIPE][CONCATENATE] sheet='{sheet}', output_column='{output_column}', input_columns={input_columns}, row_count={len(rows)}")
    
    for row in rows:
        pieces = []
        
        for col_name in input_columns:
            raw = row.get(col_name, "")
            if raw is None:
                raw = ""
            text = str(raw).strip()
            if text:
                pieces.append(text)
        
        if not pieces:
            result = ""
        else:
            result = " ".join(pieces)
        
        row[output_column] = result
    
    work[sheet] = rows
    print(f"[RECIPE][CONCATENATE] completed sheet='{sheet}', output_column='{output_column}', rows_updated={len(rows)}")


def assign_other(
    work: Dict[str, List[Dict[str, Any]]],
    sheet_name: str,
    group_column: str,
    mappings: List[Tuple[str, str]],
    errors: List[str],
) -> None:
    """
    Assign "colleague" information to each row in a sheet, based on grouping and rotating across colleagues in that group.
    
    Syntax examples:
        Assign other - (Contacts output, Website, First Name:Other First | Last Name:Other Last)
        Assign other - (Contacts output, Website, First Name:Other First)
    
    Behavior:
    1. Group rows by GroupColumn value (case-sensitive, treating None as "").
    2. For each row, choose a different colleague in the same group.
    3. Copy values from the colleague's Source columns into the current row's Dest columns.
    4. Enforce that the primary mapping (first mapping) has different values between current and colleague.
    5. Use deterministic rotation to select colleagues.
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet_name: Name of sheet in work to process
        group_column: Column name to group rows by
        mappings: List of (source_col, dest_col) tuples for column mappings
        errors: List to append error messages to
    """
    # Check sheet existence
    if sheet_name not in work:
        print(f"[RECIPE][ASSIGN_OTHER][WARN] sheet='{sheet_name}' not found in work; available_sheets={list(work.keys())}")
        errors.append(f"Assign other: sheet '{sheet_name}' not found")
        return
    
    # Load rows
    rows = work[sheet_name]
    
    if not rows:
        print(f"[RECIPE][ASSIGN_OTHER] sheet='{sheet_name}' has no rows; nothing to assign")
        return
    
    # Log start
    print(f"[RECIPE][ASSIGN_OTHER] sheet='{sheet_name}', group_column='{group_column}', mappings={mappings}, total_rows={len(rows)}")
    
    # Determine primary mapping (first mapping)
    primary_source_col, primary_dest_col = mappings[0]
    
    # Build grouping: group_value -> list of row indices
    groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        group_value = str(row.get(group_column, "") or "")
        groups.setdefault(group_value, []).append(idx)
    
    # Process each group independently
    for group_value, group_indices in groups.items():
        print(f"[RECIPE][ASSIGN_OTHER] group='{group_value}', size={len(group_indices)}")
        
        # For each row in the group (by relative position k)
        for k, row_idx in enumerate(group_indices):
            current_row = rows[row_idx]
            
            # Build candidate_indices
            candidate_indices = []
            
            for candidate_idx in group_indices:
                # Skip self
                if candidate_idx == row_idx:
                    continue
                
                candidate_row = rows[candidate_idx]
                
                # Check primary mapping inequality
                curr_val = str(current_row.get(primary_source_col, "") or "").strip()
                other_val = str(candidate_row.get(primary_source_col, "") or "").strip()
                
                # If both are non-empty and equal, skip this candidate
                if curr_val and other_val and curr_val == other_val:
                    continue
                
                # Check at least one non-empty mapping source
                has_non_empty = False
                for source_col, _ in mappings:
                    val = str(candidate_row.get(source_col, "") or "").strip()
                    if val:
                        has_non_empty = True
                        break
                
                if not has_non_empty:
                    continue
                
                # Candidate is valid
                candidate_indices.append(candidate_idx)
            
            # If no eligible candidates, skip this row
            if not candidate_indices:
                continue
            
            # Choose candidate with rotation
            chosen_idx = candidate_indices[k % len(candidate_indices)]
            chosen_row = rows[chosen_idx]
            
            # Write values from chosen colleague to current row
            for source_col, dest_col in mappings:
                value = chosen_row.get(source_col, "")
                current_row[dest_col] = value
    
    # Log completion
    print(f"[RECIPE][ASSIGN_OTHER] completed sheet='{sheet_name}'")


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


def copy_by_key(
    work: Dict[str, List[Dict[str, Any]]],
    source_sheet: str,
    target_sheet: str,
    source_key_column: str,
    target_key_column: str,
    errors: List[str],
) -> None:
    """
    Copy ALL columns from source sheet row into target sheet row, matched by key.
    
    For each row in TargetSheet:
    - Get target_key = row[TargetKeyColumn]
    - Find a matching row in SourceSheet where SourceKeyColumn == target_key
    - Copy ALL columns from that source row into the target row (creating columns if needed)
    - If there is no match for that key, do nothing for that row (leave existing values as-is).
    
    Example:
        Copy by key - (Acct, Contacts output, Website, Website)
        - For each row in "Contacts output":
          - Use Website as the key
          - Look up the row in "Acct" with the same Website
          - Copy ALL columns from the source row to the target row
          - Create any missing columns in "Contacts output"
          - Overwrite values in matching columns with the source row values
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        source_sheet: Name of source sheet in work
        target_sheet: Name of target sheet in work
        source_key_column: Column name in source sheet to use as lookup key
        target_key_column: Column name in target sheet to use as lookup key
        errors: List to append error messages to (mutated)
    """
    # Step 1: Check sheets exist
    if source_sheet not in work or target_sheet not in work:
        error_msg = f"[RECIPE][COPY_BY_KEY][ERROR] source_sheet='{source_sheet}' or target_sheet='{target_sheet}' not found; available sheets={list(work.keys())}"
        print(error_msg)
        errors.append(f"Copy by key: missing sheet '{source_sheet}' or '{target_sheet}'")
        return
    
    # Step 2: Fetch row lists
    source_rows = work[source_sheet]
    target_rows = work[target_sheet]
    
    # Step 3: Build lookup dict (from source sheet)
    lookup: Dict[str, Dict[str, Any]] = {}
    
    for row in source_rows:
        key_raw = row.get(source_key_column, "")
        if key_raw is None:
            key_raw = ""
        key = str(key_raw).strip()
        
        # Skip rows with empty key
        if not key:
            continue
        
        # Store mapping (last source row with the same key wins)
        lookup[key] = row
    
    # Step 4: Log start
    print(f"[RECIPE][COPY_BY_KEY] source_sheet='{source_sheet}', target_sheet='{target_sheet}', source_key_column='{source_key_column}', target_key_column='{target_key_column}', source_rows={len(source_rows)}, target_rows={len(target_rows)}, distinct_keys={len(lookup)}")
    
    # Step 5: For each row in target sheet
    for idx, tgt_row in enumerate(target_rows):
        tgt_key_raw = tgt_row.get(target_key_column, "")
        if tgt_key_raw is None:
            tgt_key_raw = ""
        tgt_key = str(tgt_key_raw).strip()
        
        # Skip rows with empty target key
        if not tgt_key:
            continue
        
        # Skip if no match found
        if tgt_key not in lookup:
            continue
        
        # Copy ALL columns from source row to target row
        src_row = lookup[tgt_key]
        for col_name, value in src_row.items():
            tgt_row[col_name] = value
    
    # Step 6: Assign back (work is already mutated, but be explicit)
    work[target_sheet] = target_rows
    
    # Step 7: Log completion
    print(f"[RECIPE][COPY_BY_KEY] completed source_sheet='{source_sheet}' -> target_sheet='{target_sheet}'")


def insert_column(work: Dict[str, List[Dict[str, Any]]],
                   sheet: str,
                   column_name: str,
                   errors: List[str]) -> None:
    """
    Ensure that a given column exists on a given sheet.
    
    If the column doesn't exist, it is created and set to "" (empty string) for every row.
    If it does exist, existing values are left as-is, but missing row keys are filled in as empty strings.
    
    This task does NOT:
    - Delete columns
    - Move column positions
    - Change any values except to add "" where the column didn't exist
    
    Example:
        Insert column - (Contacts output, Situational)
        - Ensures that "Situational" exists on "Contacts output"
        - If it doesn't exist, creates it and sets "" for every row
        - If it already exists, leaves values as-is and backfills missing keys as ""
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet in work
        column_name: Name of column to ensure exists
        errors: List to append error messages to
    """
    print(f"[RECIPE][INSERT_COLUMN] sheet='{sheet}', col='{column_name}'")
    
    # Step 1: Check if sheet exists
    if sheet not in work:
        error_msg = f"sheet '{sheet}' not found"
        print(f"[RECIPE][INSERT_COLUMN][ERROR] {error_msg}")
        errors.append(error_msg)
        return
    
    rows = work[sheet]
    
    try:
        # Step 2: For each row, ensure column_name exists
        # If column_name not in row, set it to ""
        for row in rows:
            if column_name not in row:
                row[column_name] = ""
        
        print(f"[RECIPE][INSERT_COLUMN] Completed: added column '{column_name}' to sheet '{sheet}'")
    except Exception as e:
        # Step 3: Error handling - catch unexpected exceptions
        error_msg = f"Unexpected error in insert_column: {str(e)}"
        print(f"[RECIPE][INSERT_COLUMN][ERROR] {error_msg}")
        errors.append(error_msg)


def copy_columns(work: Dict[str, List[Dict[str, Any]]],
                 source_sheet: str,
                 target_sheet: str,
                 mappings: List[Tuple[str, str]],
                 errors: List[str]) -> None:
    """
    Copy columns from source sheet to target sheet, replacing entire columns.
    
    For each mapping (src_col, tgt_col):
    - Copies all values from src_col in source_sheet to tgt_col in target_sheet
    - Aligns rows by index (source row 0 → target row 0, etc.)
    - If target has more rows than source, clears cells beyond source's last row (sets to "")
    - If target has fewer rows than source, extends target rows to match source length
    - Completely replaces the target column's data (all rows overwritten)
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        source_sheet: Name of source sheet in work
        target_sheet: Name of target sheet in work
        mappings: List of (source_column, target_column) tuples
        errors: List to append error messages to (mutated)
    """
    print(f"[RECIPE][COPY_COLUMNS] Source sheet: {source_sheet}, Target sheet: {target_sheet}")
    print(f"[RECIPE][COPY_COLUMNS] Mappings: {mappings}")
    
    # Check if source and target sheets exist
    # If either is missing, log error, append to errors, and return WITHOUT mutating work
    if source_sheet not in work or target_sheet not in work:
        error_msg = f"[RECIPE][COPY_COLUMNS][ERROR] source_sheet='{source_sheet}' or target_sheet='{target_sheet}' not found; available sheets={list(work.keys())}"
        print(error_msg)
        errors.append(error_msg)
        return  # Do NOT mutate work
    
    source_rows = work[source_sheet]
    target_rows = work[target_sheet]
    
    source_len = len(source_rows)
    target_len = len(target_rows)
    total_len = max(source_len, target_len)
    
    print(f"[RECIPE][COPY_COLUMNS] Source rows: {source_len}, Target rows: {target_len}, Total length: {total_len}")
    
    # Ensure target_rows has length total_len (extend if needed)
    while len(target_rows) < total_len:
        target_rows.append({})
    
    # Process each mapping
    for src_col, tgt_col in mappings:
        print(f"[RECIPE][COPY_COLUMNS] Copying '{src_col}' → '{tgt_col}'")
        
        # Copy values row by row
        for i in range(total_len):
            # Ensure target_rows[i] is a dict
            if not isinstance(target_rows[i], dict):
                target_rows[i] = {}
            
            # Get value from source (or "" if beyond source length or column missing)
            if i < source_len:
                value = source_rows[i].get(src_col, "")
            else:
                value = ""  # Clear cells beyond source's last row
            
            # Set target column value
            target_rows[i][tgt_col] = value
        
        print(f"[RECIPE][COPY_COLUMNS] Copied '{src_col}' → '{tgt_col}' for {total_len} rows")
    
    print(f"[RECIPE][COPY_COLUMNS] Completed copying {len(mappings)} column(s)")


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
    input_sheet: str,
    output_sheet: str,
    output_column: str,
    model: str,
    prompt_template: str,
    condition_column: Optional[str],
    condition_substring: Optional[str],
    errors: List[str],
    ai_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Execute an AI recipe task with optional conditional execution.
    
    This function:
    1. Validates sheet existence
    2. For each row, checks idempotence (skip if output cell is non-empty)
    3. For each row, evaluates condition if present (skips row if condition is false)
    4. Builds prompts for eligible rows by substituting {ColumnName} placeholders
    5. Calls AI API with batching, concurrency limits, and retries
    6. Writes results to output column in output sheet
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        input_sheet: Name of input sheet
        output_sheet: Name of output sheet
        output_column: Name of column to write results to
        model: Model identifier string (e.g., "openai/gpt-4o-mini")
        prompt_template: Prompt template with {ColumnName} placeholders
        condition_column: Optional column name for WHEN condition
        condition_substring: Optional substring for WHEN condition
        errors: List to append error messages to
    """
    # Validate sheet existence
    if input_sheet not in work:
        error_msg = f"[RECIPE][AI][ERROR] input_sheet='{input_sheet}' not found; available_sheets={list(work.keys())}"
        print(error_msg)
        errors.append(error_msg)
        return
    
    if output_sheet not in work:
        error_msg = f"[RECIPE][AI][ERROR] output_sheet='{output_sheet}' not found; available_sheets={list(work.keys())}"
        print(error_msg)
        errors.append(error_msg)
        return
    
    input_rows = work[input_sheet]
    output_rows = work[output_sheet]
    
    # Ensure row alignment: if input and output are different sheets,
    # we assume they have the same row ordering (row i in input corresponds to row i in output)
    # If they're the same sheet, they refer to the same list
    if input_sheet == output_sheet:
        output_rows = input_rows
    else:
        # Ensure output_rows has at least as many rows as input_rows
        # If output has fewer rows, extend it with empty dicts
        while len(output_rows) < len(input_rows):
            output_rows.append({})
    
    # Map model name to OpenRouter identifier
    try:
        mapped_model = _map_model_name(model)
    except ValueError as e:
        error_msg = f"[RECIPE][AI][ERROR] model='{model}' is invalid: {str(e)}"
        print(error_msg)
        errors.append(error_msg)
        return
    
    # Determine column headers from first row (if available)
    # We'll use all keys in the row as potential column names
    column_headers = set()
    if len(input_rows) > 0:
        column_headers = set(input_rows[0].keys())
    
    # Track statistics
    total_rows = len(input_rows)
    eligible_count = 0
    successful_count = 0
    failed_count = 0
    
    # Log start
    condition_info = ""
    if condition_column and condition_substring:
        condition_info = f", condition_column='{condition_column}', condition_substring='{condition_substring}'"
    print(f"[RECIPE][AI] input_sheet='{input_sheet}', output_sheet='{output_sheet}', output_column='{output_column}', model='{model}', total_rows={total_rows}{condition_info}")
    
    # Helper function to evaluate condition for a row
    def evaluate_condition(row: Dict[str, Any]) -> bool:
        """
        Evaluate the condition for a row.
        
        Returns:
            True if condition is None (unconditional) or if condition evaluates to True.
            False if condition evaluates to False.
        """
        if condition_column is None or condition_substring is None:
            return True  # Unconditional: process all rows
        
        # Get condition column value
        cond_raw = row.get(condition_column, "")
        cond_str = str(cond_raw).strip()
        
        # Case-insensitive contains check
        return condition_substring.lower() in cond_str.lower()
    
    # Build prompts for each row
    async def process_row(row_index: int, in_row: Dict[str, Any], out_row: Dict[str, Any]) -> None:
        """Process a single row: check idempotence, check condition, build prompt, call AI, write result."""
        nonlocal eligible_count, successful_count, failed_count
        
        try:
            # 1. Check idempotence: if output cell is already non-empty, skip
            out_raw = out_row.get(output_column, "")
            out_str = str(out_raw).strip()
            if out_str != "":
                # Already filled, skip (idempotence)
                return
            
            # 2. Check condition: if condition is false, skip this row
            if not evaluate_condition(in_row):
                # Condition is false: skip this row, leave output cell unchanged
                return
            
            # Row is eligible for processing
            eligible_count += 1
            
            # 3. Build row values dict (column name -> value) from input row
            variables = {}
            for col_name in column_headers:
                value = in_row.get(col_name)
                # Convert to string, handling None/empty
                variables[col_name] = str(value) if value is not None else ""
            
            # 4. Substitute placeholders in prompt template
            # Pattern: {ColumnName} where ColumnName matches a column header exactly
            # If a column is referenced but doesn't exist, substitute empty string
            prompt = prompt_template
            
            # Find all placeholders in the prompt (pattern: {ColumnName})
            placeholder_pattern = r'\{([^}]+)\}'
            placeholders = re.findall(placeholder_pattern, prompt)
            
            # Replace each placeholder with the corresponding value (or empty string if not found)
            for placeholder_name in placeholders:
                value = variables.get(placeholder_name, "")
                prompt = prompt.replace(f"{{{placeholder_name}}}", value)
            
            # 5. Call AI with semaphore for concurrency control
            async with semaphore:
                ai_response = await _call_ai_with_retry(ai_client, prompt, mapped_model)
            
            # 6. Write result to output column
            if ai_response is None:
                # AI call failed: leave output cell unchanged (likely empty)
                failed_count += 1
                error_msg = f"[RECIPE][AI][ERROR] Row {row_index}: AI call failed"
                print(error_msg)
            else:
                out_row[output_column] = ai_response
                successful_count += 1
        except Exception as e:
            # Per-row error: leave output cell unchanged (likely empty)
            failed_count += 1
            error_msg = f"[RECIPE][AI][ERROR] Row {row_index}: {str(e)}"
            print(error_msg)
            # Don't append to errors list for per-row errors to avoid spam
    
    # Process all rows concurrently (with semaphore limiting concurrency)
    # Batch processing: process in chunks to avoid overwhelming the system
    batch_size = 50
    for batch_start in range(0, len(input_rows), batch_size):
        batch_end = min(batch_start + batch_size, len(input_rows))
        
        # Process batch concurrently
        tasks = [
            process_row(batch_start + i, input_rows[batch_start + i], output_rows[batch_start + i])
            for i in range(batch_end - batch_start)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Replace updated rows back into work (output_rows is already a reference, but ensure it's updated)
    work[output_sheet] = output_rows
    
    # Log summary
    print(f"[RECIPE][AI] completed input_sheet='{input_sheet}', output_sheet='{output_sheet}', model='{model}', total_rows={total_rows}, eligible={eligible_count}, successful={successful_count}, failed={failed_count}")


async def _call_exa_with_retry(
    exa_client: Exa,
    website: str,
    max_retries: int = 3,
    backoff_delays: List[float] = None
) -> Optional[str]:
    """
    Call Exa API with retry logic and exponential backoff.
    
    Args:
        exa_client: Exa client instance
        website: The website URL to summarize
        max_retries: Maximum number of retry attempts
        backoff_delays: List of delay seconds for each retry (default: [1, 2])
        
    Returns:
        Summary text or None if all attempts fail
    """
    if backoff_delays is None:
        backoff_delays = [1, 2]
    
    # Normalize website URL
    normalized_website = website.strip()
    if not normalized_website:
        return None
    
    # Ensure website has protocol
    if not normalized_website.startswith(('http://', 'https://')):
        normalized_website = f"https://{normalized_website}"
    
    for attempt in range(max_retries):
        try:
            # Call Exa.AI to get summary
            # Note: Exa client is synchronous, so we run it in a thread to avoid blocking
            response = await asyncio.to_thread(
                exa_client.search_and_contents,
                query=f"company information about {normalized_website}",
                contents={
                    "text": {"max_characters": 1000}
                },
                num_results=1
            )
            
            if response.results and len(response.results) > 0:
                # Extract text content from the first result
                result = response.results[0]
                if hasattr(result, 'text') and result.text:
                    return result.text
                elif hasattr(result, 'id'):
                    # If no text, try to get content from ID
                    content_response = await asyncio.to_thread(
                        exa_client.get_contents,
                        ids=[result.id],
                        text={"max_characters": 1000}
                    )
                    if content_response.results and len(content_response.results) > 0:
                        return content_response.results[0].text
            
            return None
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                await asyncio.sleep(delay)
            else:
                # Last attempt failed
                return None
    
    return None


async def run_exa_task(
    work: Dict[str, List[Dict[str, Any]]],
    sheet: str,
    website_column: str,
    output_column: str,
    reference_column: Optional[str],
    exa_client: Exa,
    semaphore: asyncio.Semaphore,
    errors: List[str]
) -> None:
    """
    Execute an Exa recipe task with optional reference column gating.
    
    This function:
    1. Reads input sheet data
    2. For each row, checks reference_column if provided (only runs when reference column IS blank)
    3. Checks if website is non-empty and output column is empty (idempotent)
    4. Calls Exa API with batching, concurrency limits, and retries
    5. Writes results to output column
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet: Name of sheet to process
        website_column: Name of column containing website URLs
        output_column: Name of column to write results to
        reference_column: Optional column name; if provided, only process rows where this column IS blank
        exa_client: Exa client instance
        semaphore: Semaphore to limit concurrent Exa requests
        errors: List to append error messages to (mutated)
    """
    # Resolve sheet
    rows = work.get(sheet)
    if rows is None:
        available_sheets = list(work.keys())
        error_msg = f"[RECIPE][EXA][ERROR] sheet='{sheet}' not found; available_sheets={available_sheets}"
        print(error_msg)
        errors.append(error_msg)
        return
    
    # If rows is empty, log and return
    if len(rows) == 0:
        print(f"[RECIPE][EXA] sheet='{sheet}' has no rows; nothing to do")
        return
    
    # Ensure output_column exists in all rows
    for row in rows:
        if output_column not in row:
            row[output_column] = ""
    
    # Count eligible rows (for logging)
    eligible_rows = []
    total_rows = len(rows)
    
    for row_index, row in enumerate(rows):
        # Get website value
        website_raw = row.get(website_column, "")
        website_str = str(website_raw).strip()
        
        # Skip if website is empty
        if not website_str:
            continue
        
        # Check reference column if provided
        if reference_column is not None:
            ref_raw = row.get(reference_column, "")
            ref_str = str(ref_raw).strip()
            # Skip if reference column is NOT empty (we only run when it IS blank)
            if ref_str:
                continue
        
        # Check if output column already has a non-empty value (idempotent)
        current_output = row.get(output_column, "")
        current_output_str = str(current_output).strip()
        if current_output_str:
            # Output already populated, skip to avoid re-calling Exa
            continue
        
        # Row is eligible for Exa processing
        eligible_rows.append((row_index, row, website_str))
    
    eligible_count = len(eligible_rows)
    
    # Log start
    ref_col_str = f", reference_column='{reference_column}'" if reference_column else ""
    print(f"[RECIPE][EXA] sheet='{sheet}', website_column='{website_column}', output_column='{output_column}'{ref_col_str}, total_rows={total_rows}, eligible_rows={eligible_count}")
    
    if eligible_count == 0:
        print(f"[RECIPE][EXA] sheet='{sheet}', website_column='{website_column}', output_column='{output_column}'{ref_col_str}, eligible_rows=0")
        return
    
    # Process each eligible row
    successful = 0
    failed = 0
    
    async def process_row(row_index: int, row: Dict[str, Any], website: str) -> None:
        """Process a single row: call Exa, write result."""
        nonlocal successful, failed
        try:
            # Call Exa with semaphore for concurrency control
            async with semaphore:
                exa_summary = await _call_exa_with_retry(exa_client, website)
            
            # Write result to output column
            if exa_summary is None:
                # Exa call failed, leave empty
                failed += 1
                error_msg = f"[RECIPE][EXA][ERROR] Exa request failed for website='{website}'"
                print(error_msg)
                errors.append(error_msg)
            else:
                row[output_column] = exa_summary
                successful += 1
                
        except Exception as e:
            # Per-row error: leave output empty and continue processing other rows
            failed += 1
            error_msg = f"[RECIPE][EXA][ERROR] Exa request failed for website='{website}': {str(e)}"
            print(error_msg)
            errors.append(error_msg)
            row[output_column] = ""
    
    # Process all eligible rows concurrently (with semaphore limiting concurrency)
    # Batch processing: process in chunks to avoid overwhelming the system
    batch_size = 50
    for batch_start in range(0, len(eligible_rows), batch_size):
        batch_end = min(batch_start + batch_size, len(eligible_rows))
        batch_data = eligible_rows[batch_start:batch_end]
        
        # Process batch concurrently
        tasks = [
            process_row(row_index, row, website)
            for row_index, row, website in batch_data
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Log summary
    print(f"[RECIPE][EXA] completed sheet='{sheet}', eligible={eligible_count}, successful={successful}, failed={failed}")


def run_recipe(project_id: str,
               run_id: str,
               work: Dict[str, List[Dict[str, Any]]],
               progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Main entry point for the recipe engine.
    
    This function:
    1. Parses tasks from Master sheet in work dictionary
    2. Validates that all sheets referenced by tasks exist
    3. Executes tasks in order on in-memory data
    4. Returns results without performing any I/O
    
    Args:
        project_id: Project ID (for logging only, not used for I/O)
        run_id: Run ID (for logging only, not used for I/O)
        work: Dictionary mapping sheet names to their row lists (all sheets from spreadsheet)
        progress_callback: Optional callback function(row_index: int, status: str) called after each task completes
        
    Returns:
        Dictionary with keys:
        - ok: bool (True if successful, False if errors)
        - errors: List[str] (error messages, empty if ok=True)
        - urls_output: List[Dict[str, Any]] or None (final URLs output rows)
        - contacts_output: List[Dict[str, Any]] or None (final Contacts output rows)
        - master_status_updates: List[Dict[str, int]] or None (list of {row_index, status} dicts)
    """
    print(f"[RECIPE][RUN] initial work sheets: {list(work.keys())}")
    
    # Guard rail: If work dict is empty, this is a hard error
    if not work:
        error_msg = "[RECIPE] No sheets loaded into work dictionary. Spreadsheet appears to be empty."
        print(f"[RECIPE][RUN] ERROR: {error_msg}")
        return {
            "ok": False,
            "errors": [error_msg],
            "urls_output": None,
            "contacts_output": None,
            "master_status_updates": None,
        }
    
    # Extract Master sheet for task parsing
    master_rows = work.get("Master", [])
    if not master_rows:
        error_msg = "[RECIPE] Master tab is missing or empty. Cannot run recipe without task definitions."
        print(f"[RECIPE][RUN] ERROR: {error_msg}")
        return {
            "ok": False,
            "errors": [error_msg],
            "urls_output": None,
            "contacts_output": None,
            "master_status_updates": None,
        }
    
    # Build inputs dictionary (read-only source for COPY_SHEET tasks)
    # Only include input sheets that exist in work
    inputs: Dict[str, List[Dict[str, Any]]] = {}
    for sheet_name in INPUT_SHEETS:
        if sheet_name in work:
            inputs[sheet_name] = work[sheet_name]
    
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
    
    # Guard rail: Check if Master tab has zero runnable tasks
    if not tasks or len(tasks) == 0:
        error_msg = "[RECIPE] Master tab contains zero runnable tasks. Cannot execute recipe."
        print(f"[RECIPE][RUN] ERROR: {error_msg}")
        return {
            "ok": False,
            "errors": [error_msg],
            "urls_output": None,
            "contacts_output": None,
            "master_status_updates": None,
        }
    
    # Validate that all sheets referenced by tasks exist in work dictionary
    referenced_sheets = _extract_referenced_sheets(tasks)
    missing_sheets = []
    errors = []
    for sheet_name in referenced_sheets:
        if sheet_name not in work:
            missing_sheets.append(sheet_name)
            error_msg = f"[RECIPE][ERROR] sheet '{sheet_name}' not found in work; available sheets={list(work.keys())}"
            print(f"[RECIPE][RUN] {error_msg}")
            errors.append(error_msg)
    
    if missing_sheets:
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
    
    # Initialize Exa client and semaphore (only if we have Exa tasks)
    exa_client = None
    exa_semaphore = None
    has_exa_tasks = any(task.type == TASK_EXA for task in tasks)
    
    if has_exa_tasks:
        # Initialize Exa client (same pattern as enrich_workflow)
        exa_api_key = os.getenv("EXA_API_KEY")
        if not exa_api_key:
            return {
                "ok": False,
                "errors": ["EXA_API_KEY environment variable is required for Exa tasks"],
                "urls_output": None,
                "contacts_output": None,
                "master_status_updates": None,
            }
        
        exa_client = Exa(api_key=exa_api_key)
        
        # Create semaphore for concurrency control (5 concurrent requests, same as enrichment)
        exa_semaphore = asyncio.Semaphore(5)
    
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
                print(f"[RECIPE][NORMALIZE_URLS] Executing task at row {task.row_index}")
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
            elif task.type == TASK_FILTER_BLANK:
                filter_blank(
                    work,
                    task.params["sheet"],
                    task.params["column"]
                )
            elif task.type == TASK_FILTER_MATCH:
                filter_match(
                    work,
                    task.params["source_sheet"],
                    task.params["source_column"],
                    task.params["lookup_sheet"],
                    task.params["lookup_column"],
                    errors
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
            elif task.type == TASK_COUNT_MATCHES:
                count_matches(
                    work,
                    task.params["source_sheet"],
                    task.params["group_column"],
                    task.params["target_sheet"],
                    task.params["target_column"],
                    task.params["count_column"]
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
            elif task.type == TASK_REMOVE_TEXT:
                remove_text(
                    work,
                    task.params["sheet"],
                    task.params["column"],
                    task.params["phrases"],
                    errors,
                )
            elif task.type == TASK_CONCATENATE:
                concatenate(
                    work,
                    task.params["sheet"],
                    task.params["output_column"],
                    task.params["input_columns"],
                    errors,
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
                assign_other(
                    work,
                    task.params["sheet"],
                    task.params["group_column"],
                    task.params["mappings"],
                    errors,
                )
            elif task.type == TASK_COPY_BY_KEY:
                copy_by_key(
                    work,
                    task.params["source_sheet"],
                    task.params["target_sheet"],
                    task.params["source_key_column"],
                    task.params["target_key_column"],
                    errors,
                )
            elif task.type == TASK_INSERT_COLUMN:
                insert_column(
                    work,
                    task.params["sheet"],
                    task.params["column"],
                    errors
                )
            elif task.type == TASK_COPY_COLUMNS:
                print(f"[RECIPE][COPY_COLUMNS] Executing task at row {task.row_index}")
                copy_columns(
                    work,
                    task.params["source_sheet"],
                    task.params["target_sheet"],
                    task.params["mappings"],
                    errors
                )
            elif task.type == TASK_AI:
                # AI tasks are async, so we need to run them in an event loop
                # Since run_recipe is sync, we use asyncio.run()
                # Note: asyncio.run() creates a new event loop, so this is safe even if
                # called from a sync context (which is the case in worker.py)
                try:
                    asyncio.run(run_ai_task(
                        work,
                        task.params["input_sheet"],
                        task.params["output_sheet"],
                        task.params["output_column"],
                        task.params["model"],
                        task.params["prompt_template"],
                        task.params.get("condition_column"),
                        task.params.get("condition_substring"),
                        errors,
                        ai_client,
                        ai_semaphore,
                    ))
                except Exception as e:
                    error_msg = f"[RECIPE][AI][ERROR] AI task failed on row {task.row_index}: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)
            elif task.type == TASK_EXA:
                # Exa tasks are async, so we need to run them in an event loop
                # Since run_recipe is sync, we use asyncio.run()
                # Note: asyncio.run() creates a new event loop, so this is safe even if
                # called from a sync context (which is the case in worker.py)
                try:
                    asyncio.run(run_exa_task(
                        work,
                        task.params["sheet"],
                        task.params["website_column"],
                        task.params["output_column"],
                        task.params.get("reference_column"),
                        exa_client,
                        exa_semaphore,
                        errors
                    ))
                except Exception as e:
                    error_msg = f"[RECIPE][EXA][ERROR] Exa task failed on row {task.row_index}: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)
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
    
    # Check if there were any runtime errors (e.g., from insert_column)
    if errors:
        return {
            "ok": False,
            "errors": errors,
            "urls_output": None,
            "contacts_output": None,
            "master_status_updates": None,
        }
    
    return {
        "ok": True,
        "errors": [],
        "urls_output": urls_output_rows,
        "contacts_output": contacts_output_rows,
        "master_status_updates": master_status_updates,
    }

