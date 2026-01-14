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
TASK_AI = "AI"
TASK_EXA = "EXA"

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
    Parse 'Remove text - <SheetName> | <ColumnName> | "<Phrase1>" | "<Phrase2>" | ...' pattern.
    
    Syntax examples:
        Remove text - URLs output | Company Name | "LLC" | "Inc." | "Corporation"
        Remove text - Contacts output | Title | " at {Company}" | " (Remote)" | " (Contract)"
    
    Returns:
        Dict with 'sheet_name', 'column_name', and 'phrases' (list[str]) keys, or None if parse fails
    """
    # Check if task name starts with "Remove text -" (case-insensitive)
    if not task_name.lower().startswith("remove text -"):
        return None
    
    # Remove the prefix and trim
    prefix_len = len("Remove text -")
    rest = task_name[prefix_len:].strip()
    
    if not rest:
        return None
    
    # Split by pipe character
    tokens = [token.strip() for token in rest.split("|")]
    
    # Need at least 3 tokens: sheet_name, column_name, and at least one phrase
    if len(tokens) < 3:
        return None
    
    sheet_name = tokens[0]
    column_name = tokens[1]
    
    # Validate that sheet_name and column_name are non-empty
    if not sheet_name or not column_name:
        return None
    
    # Parse phrases (tokens[2:] onwards)
    phrases = []
    for token in tokens[2:]:
        if not token:
            continue  # Skip empty tokens
        
        # Trim whitespace
        phrase = token.strip()
        
        # Optionally strip surrounding double quotes if present
        if phrase.startswith('"') and phrase.endswith('"') and len(phrase) >= 2:
            phrase = phrase[1:-1].strip()
        
        # Ignore empty phrases after trimming
        if phrase:
            phrases.append(phrase)
    
    # Must have at least one phrase
    if not phrases:
        return None
    
    return {
        "sheet_name": sheet_name,
        "column_name": column_name,
        "phrases": phrases
    }


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


def _parse_copy_by_key(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Copy by key - <SourceSheet> | <SourceKeyColumn> | <TargetSheet> | <TargetKeyColumn>' pattern.
    
    Syntax examples:
        Copy by key - URLs output | Website | Contacts output | Website
        Copy by key - Accounts output | Domain | Leads output | WebsiteDomain
    
    Returns:
        Dict with 'source_sheet_name', 'source_key_column', 'target_sheet_name', and 'target_key_column' keys, or None if parse fails
    """
    # Check if task name starts with "Copy by key -" (case-insensitive)
    if not task_name.lower().startswith("copy by key -"):
        return None
    
    # Remove the prefix and trim
    prefix_len = len("Copy by key -")
    rest = task_name[prefix_len:].strip()
    
    if not rest:
        return None
    
    # Split by pipe character
    tokens = [token.strip() for token in rest.split("|")]
    
    # Must have exactly 4 tokens
    if len(tokens) != 4:
        return None
    
    source_sheet_name = tokens[0]
    source_key_column = tokens[1]
    target_sheet_name = tokens[2]
    target_key_column = tokens[3]
    
    # Validate that all tokens are non-empty
    if not source_sheet_name or not source_key_column or not target_sheet_name or not target_key_column:
        return None
    
    return {
        "source_sheet_name": source_sheet_name,
        "source_key_column": source_key_column,
        "target_sheet_name": target_sheet_name,
        "target_key_column": target_key_column
    }


def _parse_insert_column(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'Insert column - <SheetName> | <ColumnName>' pattern.
    
    Syntax examples:
        Insert column - Contacts output | GPT_Notes
        Insert column - URLs output | Segment Override
    
    Returns:
        Dict with 'sheet_name' and 'column_name' keys, or None if parse fails
    """
    # Check if task name starts with "Insert column -" (case-insensitive)
    if not task_name.lower().startswith("insert column -"):
        return None
    
    # Remove the prefix and trim
    prefix_len = len("Insert column -")
    rest = task_name[prefix_len:].strip()
    
    if not rest:
        return None
    
    # Split by pipe character
    tokens = [token.strip() for token in rest.split("|")]
    
    # Must have exactly 2 tokens
    if len(tokens) != 2:
        return None
    
    sheet_name = tokens[0]
    column_name = tokens[1]
    
    # Validate that all tokens are non-empty
    if not sheet_name or not column_name:
        return None
    
    return {
        "sheet_name": sheet_name,
        "column_name": column_name
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
    Parse a WHEN condition segment: 'WHEN {ColumnName} contains: "Some Text"'
    
    Args:
        condition_segment: The condition segment string (e.g., 'WHEN {Segment} contains: "B2B"')
        
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
    
    # After placeholder, expect space and "contains:" (case-insensitive)
    after_placeholder = after_when[placeholder_end + 1:].strip()
    after_placeholder_lower = after_placeholder.lower()
    if not after_placeholder_lower.startswith("contains:"):
        return None
    
    # Find the position of "contains:" in the original string (case-insensitive)
    # We know it starts at position 0 after trimming, so find the length
    contains_len = len("contains:")
    # Remove "contains:" prefix and trim (use the actual substring from original)
    after_contains = after_placeholder[contains_len:].strip()
    if not after_contains:
        return None
    
    # Expect quoted string
    if not after_contains.startswith('"') or not after_contains.endswith('"'):
        return None
    
    if len(after_contains) < 2:
        return None
    
    # Extract substring from quotes
    substring = after_contains[1:-1].strip()
    
    return {
        "column_name": column_name,
        "substring": substring
    }


def _parse_ai_task(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse AI task pattern with optional WHEN condition.
    
    Supports two formats:
    1. Unconditional: 'AI - <InputSheet> | <OutputSheet> | <OutputColumn> | <ModelName> | """<PromptText>"""'
    2. Conditional: 'AI - <InputSheet> | <OutputSheet> | <OutputColumn> | <ModelName> | WHEN {ColumnName} contains: "Text" | """<PromptText>"""'
    
    Syntax examples:
        Example 1 (unconditional):
            AI - URLs output | URLs output | GPT_Company_Summary | gpt-mini | \"\"\"
            #Summarize what this company does in one sentence.
            
            #Company: {Company}
            #Website: {Website}
            #Short description: {Short Description}
            
            #Return a single plain-English sentence with no fluff.
            \"\"\"
        
        Example 2 (conditional):
            AI - URLs output | URLs output | GPT_Company_Summary | gpt-mini | WHEN {Segment} contains: "B2B" | \"\"\"
            #Summarize what this company does in one sentence.
            
            #Company: {Company}
            #Website: {Website}
            #Short description: {Short Description}
            
            #Return a single plain-English sentence with no fluff.
            \"\"\"
    
    Returns:
        Dict with 'input_sheet_name', 'output_sheet_name', 'output_column_name', 'model_name', 'prompt_template',
        and optionally 'condition' (None if unconditional, or dict with 'column_name' and 'substring' if conditional),
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
    
    # We expect either 4 tokens (unconditional) or 5 tokens (conditional) before the prompt
    if len(tokens_before) not in (4, 5):
        return None
    
    input_sheet_name = tokens_before[0]
    output_sheet_name = tokens_before[1]
    output_column_name = tokens_before[2]
    model_name = tokens_before[3]
    
    # Validate that none are empty
    if not input_sheet_name or not output_sheet_name or not output_column_name or not model_name:
        return None
    
    # Parse condition if present (5 tokens = conditional)
    condition = None
    if len(tokens_before) == 5:
        condition_segment = tokens_before[4]
        condition = _parse_ai_condition(condition_segment)
        if condition is None:
            return None  # Invalid condition syntax
    
    # Extract prompt template by removing outer triple quotes
    if not prompt_with_quotes.startswith('"""') or not prompt_with_quotes.endswith('"""'):
        return None
    
    prompt_template = prompt_with_quotes[3:-3]  # Remove first and last 3 characters (""")
    
    result = {
        "input_sheet_name": input_sheet_name,
        "output_sheet_name": output_sheet_name,
        "output_column_name": output_column_name,
        "model_name": model_name,
        "prompt_template": prompt_template
    }
    
    # Add condition if present
    if condition is not None:
        result["condition"] = condition
    
    return result


def _parse_exa_task(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse Exa task pattern with optional WHEN condition.
    
    Supports two formats:
    1. Unconditional: 'Exa - <SheetName> | <WebsiteColumn> | <OutputColumn>'
    2. Conditional: 'Exa - <SheetName> | <WebsiteColumn> | <OutputColumn> | WHEN {ColumnName} is not empty'
    
    Syntax examples:
        Example 1 (unconditional):
            Exa - URLs output | Website | Exa Summary
        
        Example 2 (conditional):
            Exa - URLs output | Website | Exa Summary | WHEN {Short Description} is not empty
    
    Returns:
        Dict with 'sheet_name', 'website_column', 'output_column', and optionally 'condition',
        or None if parse fails
    """
    # Check if task name starts with "Exa -" (case-insensitive)
    if not task_name.lower().startswith("exa -"):
        return None
    
    # Remove the prefix and trim
    prefix_len = len("Exa -")
    rest = task_name[prefix_len:].strip()
    
    if not rest:
        return None
    
    # Split by pipe character
    tokens = [token.strip() for token in rest.split("|")]
    
    # We expect either 3 tokens (unconditional) or 4 tokens (conditional)
    if len(tokens) not in (3, 4):
        return None
    
    sheet_name = tokens[0]
    website_column = tokens[1]
    output_column = tokens[2]
    
    # Validate that required fields are non-empty
    if not sheet_name or not website_column or not output_column:
        return None
    
    # Parse condition if present (4 tokens = conditional)
    condition = None
    if len(tokens) == 4:
        condition_segment = tokens[3]
        condition = _parse_exa_condition(condition_segment)
        if condition is None:
            return None  # Invalid condition syntax
    
    result = {
        "sheet_name": sheet_name,
        "website_column": website_column,
        "output_column": output_column
    }
    
    # Add condition if present
    if condition is not None:
        result["condition"] = condition
    
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
                       TASK_FILTER_EXCLUDE, TASK_FILTER_BLANK, TASK_COUNT_BY, TASK_SORT, TASK_REMOVE_CHARACTERS, TASK_CONCATENATE):
        sheet = params.get("sheet")
        
        # Sheet must be an output sheet
        if sheet not in OUTPUT_SHEETS:
            return f"Row {row_index}: Operation on '{sheet}' is not allowed. Only output sheets (URLs output, Contacts output) can be used for non-copy operations"
    
    elif task_type == TASK_REMOVE_TEXT:
        sheet_name = params.get("sheet_name")
        column_name = params.get("column_name")
        phrases = params.get("phrases")
        
        # Validate that all required fields are non-empty
        if not sheet_name:
            return f"Row {row_index}: Remove text task sheet_name is required"
        if not column_name:
            return f"Row {row_index}: Remove text task column_name is required"
        if not phrases or not isinstance(phrases, list) or len(phrases) == 0:
            return f"Row {row_index}: Remove text task must have at least one phrase"
        
        # Sheet must be an output sheet
        if sheet_name not in OUTPUT_SHEETS:
            return f"Row {row_index}: Operation on '{sheet_name}' is not allowed. Only output sheets (URLs output, Contacts output) can be used for non-copy operations"
    
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
    
    elif task_type == TASK_COPY_BY_KEY:
        source_sheet_name = params.get("source_sheet_name")
        source_key_column = params.get("source_key_column")
        target_sheet_name = params.get("target_sheet_name")
        target_key_column = params.get("target_key_column")
        
        # Validate that all required fields are non-empty
        if not source_sheet_name:
            return f"Row {row_index}: Copy by key task source_sheet_name is required"
        if not source_key_column:
            return f"Row {row_index}: Copy by key task source_key_column is required"
        if not target_sheet_name:
            return f"Row {row_index}: Copy by key task target_sheet_name is required"
        if not target_key_column:
            return f"Row {row_index}: Copy by key task target_key_column is required"
        
        # Both sheets must be output sheets (similar to Map task)
        if source_sheet_name not in OUTPUT_SHEETS:
            return f"Row {row_index}: Source sheet '{source_sheet_name}' must be an output sheet (URLs output or Contacts output)"
        
        if target_sheet_name not in OUTPUT_SHEETS:
            return f"Row {row_index}: Target sheet '{target_sheet_name}' must be an output sheet (URLs output or Contacts output)"
    
    elif task_type == TASK_INSERT_COLUMN:
        sheet_name = params.get("sheet_name")
        column_name = params.get("column_name")
        
        # Validate that all required fields are non-empty
        if not sheet_name:
            return f"Row {row_index}: Insert column task sheet_name is required"
        if not column_name:
            return f"Row {row_index}: Insert column task column_name is required"
        
        # Sheet must be an output sheet
        if sheet_name not in OUTPUT_SHEETS:
            return f"Row {row_index}: Operation on '{sheet_name}' is not allowed. Only output sheets (URLs output, Contacts output) can be used for non-copy operations"
    
    elif task_type == TASK_AI:
        input_sheet_name = params.get("input_sheet_name")
        output_sheet_name = params.get("output_sheet_name")
        output_column_name = params.get("output_column_name")
        model_name = params.get("model_name")
        prompt_template = params.get("prompt_template")
        condition = params.get("condition")
        
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
        
        # Validate condition if present
        if condition is not None:
            condition_column_name = condition.get("column_name")
            if not condition_column_name:
                return f"Row {row_index}: AI task condition column_name is required when condition is present"
            # Note: We don't validate that the column exists at parse time,
            # since sheet headers are validated at runtime when data is present
        
        # Input sheet must exist in inputs or work (we'll check at runtime if it's in work)
        # Output sheet should be an output sheet (but we allow any sheet that exists)
        # For now, we just validate they're non-empty strings
    
    elif task_type == TASK_EXA:
        sheet_name = params.get("sheet_name")
        website_column = params.get("website_column")
        output_column = params.get("output_column")
        condition = params.get("condition")
        
        # Validate that all required fields are non-empty
        if not sheet_name:
            return f"Row {row_index}: Exa task sheet_name is required"
        if not website_column:
            return f"Row {row_index}: Exa task website_column is required"
        if not output_column:
            return f"Row {row_index}: Exa task output_column is required"
        
        # Validate condition if present
        if condition is not None:
            condition_column_name = condition.get("column_name")
            if not condition_column_name:
                return f"Row {row_index}: Exa task condition column_name is required when condition is present"
            # Note: We don't validate that the column exists at parse time,
            # since sheet headers are validated at runtime when data is present
    
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
        elif task_name.lower().startswith("ai -"):
            parsed = _parse_ai_task(task_name)
            if parsed:
                task_type = TASK_AI
                params = parsed
        elif task_name.lower().startswith("exa -"):
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


def remove_text_task(work: Dict[str, List[Dict[str, Any]]],
                     sheet_name: str,
                     column_name: str,
                     phrases: List[str]) -> None:
    """
    Remove specific phrases (substrings) from a column in-place, case-insensitively.
    
    For each row in the sheet, removes all case-insensitive occurrences of each phrase
    from the column value, then writes the cleaned value back into the same column.
    After removal, trims whitespace and optionally collapses multiple spaces.
    
    Examples:
        - Input: "Acme LLC" with phrases=["LLC"] → Output: "Acme"
        - Input: "Head of Growth at {Company}" with phrases=[" at {Company}"] → Output: "Head of Growth"
        - Input: "Business Business LLC" with phrases=["Business", "LLC"] → Output: "" (empty after trimming)
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet_name: Name of sheet in work
        column_name: Column name to clean (modified in-place)
        phrases: List of phrase strings to remove (case-insensitive)
        
    Note:
        If sheet or column doesn't exist, the function returns early without modifying data.
        This matches the behavior of other tasks that handle missing columns gracefully.
    """
    # Look up the sheet
    rows = work.get(sheet_name)
    if rows is None:
        # Sheet not found - return early without crashing
        return
    
    # Ensure the column exists (check if it appears in any row)
    if len(rows) > 0:
        column_exists = any(column_name in row for row in rows)
        if not column_exists:
            # Column not found - return early without crashing
            return
    
    # Precompile regex patterns for case-insensitive phrase removal
    # We'll use re.IGNORECASE flag for each phrase
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
        original_value = row.get(column_name)
        
        # Convert to string and trim leading/trailing whitespace
        if original_value is None:
            original_value = ""
        else:
            original_value = str(original_value).strip()
        
        # If empty after trimming, skip heavy phrase removal
        if not original_value:
            row[column_name] = ""
            continue
        
        # Apply all phrase removals
        cleaned_value = original_value
        for pattern in phrase_patterns:
            # Remove all occurrences of this phrase (case-insensitive)
            cleaned_value = pattern.sub('', cleaned_value)
        
        # Optionally collapse multiple spaces into one
        # Replace 2+ spaces with a single space
        cleaned_value = re.sub(r' +', ' ', cleaned_value)
        
        # Trim leading/trailing whitespace again after removals
        cleaned_value = cleaned_value.strip()
        
        # Write back to the same column
        row[column_name] = cleaned_value


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


def copy_by_key_task(work: Dict[str, List[Dict[str, Any]]],
                     source_sheet_name: str,
                     source_key_column: str,
                     target_sheet_name: str,
                     target_key_column: str) -> None:
    """
    Copy all non-key columns from source sheet to target sheet, matching rows by key columns.
    
    For each row in the target sheet:
    - Look up the value of target_key_column
    - Use that to find the matching row in the source sheet by source_key_column
    - For every non-key column in the source sheet:
      - Create that column on the target sheet if it doesn't exist
      - Copy the source cell value into the target row's column
    - Rows with no matching key in the source sheet are left unchanged
    
    Example:
        Copy by key - URLs output | Website | Contacts output | Website
        - For each row in "Contacts output":
          - Use Website as the key
          - Look up the row in "URLs output" with the same Website
          - Copy all columns except Website (Segment, Employee Count, Tech Stack, etc.)
          - Create any missing columns in "Contacts output"
          - Overwrite values in those columns with the source row values
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        source_sheet_name: Name of source sheet in work
        source_key_column: Column name in source sheet to use as lookup key
        target_sheet_name: Name of target sheet in work
        target_key_column: Column name in target sheet to use as lookup key
        
    Raises:
        ValueError: If source or target sheet is not found, or if required columns are missing
    """
    # Step 1: Fetch source and target sheet data
    source_rows = work.get(source_sheet_name)
    target_rows = work.get(target_sheet_name)
    
    if source_rows is None:
        raise ValueError(f"Source sheet '{source_sheet_name}' not found")
    
    if target_rows is None:
        raise ValueError(f"Target sheet '{target_sheet_name}' not found")
    
    # Step 2: Check required columns exist
    # Check if source_key_column exists in source sheet
    if len(source_rows) > 0:
        source_key_exists = any(source_key_column in row for row in source_rows)
        if not source_key_exists:
            raise ValueError(f"Source key column '{source_key_column}' not found in sheet '{source_sheet_name}'")
    
    # Check if target_key_column exists in target sheet
    if len(target_rows) > 0:
        target_key_exists = any(target_key_column in row for row in target_rows)
        if not target_key_exists:
            raise ValueError(f"Target key column '{target_key_column}' not found in sheet '{target_sheet_name}'")
    
    # Step 3: Build source lookup dictionary
    # key -> source_row (first occurrence wins if duplicates)
    source_lookup: Dict[str, Dict[str, Any]] = {}
    
    for row in source_rows:
        # Get key value and normalize: String(value).trim()
        key_raw = row.get(source_key_column)
        if key_raw is None:
            continue
        
        key = str(key_raw).strip()
        
        # Skip rows with empty key (after trimming)
        if not key:
            continue
        
        # Only add if we haven't seen this key before (first occurrence wins)
        if key not in source_lookup:
            source_lookup[key] = row
    
    # Step 4: Determine which columns to copy
    # All columns in source sheet EXCEPT source_key_column
    source_data_columns = set()
    if len(source_rows) > 0:
        # Get all column names from the first source row
        for col_name in source_rows[0].keys():
            if col_name != source_key_column:
                source_data_columns.add(col_name)
    
    # Step 5: Ensure target has all columns
    # For each source data column, add it to all target rows if it doesn't exist
    for col_name in source_data_columns:
        for row in target_rows:
            if col_name not in row:
                row[col_name] = ""
    
    # Step 6: Apply the mapping
    for row in target_rows:
        # Get target key value and normalize: String(value).trim()
        target_key_raw = row.get(target_key_column)
        if target_key_raw is None:
            continue
        
        target_key = str(target_key_raw).strip()
        
        # Skip rows with empty target key (after trimming)
        if not target_key:
            continue
        
        # Look up source row
        source_row = source_lookup.get(target_key)
        
        # If no source match, skip this row (leave values unchanged)
        if source_row is None:
            continue
        
        # If found, copy all source data columns to target row
        for col_name in source_data_columns:
            # Get value from source row (can be None, empty, or any value)
            source_value = source_row.get(col_name)
            
            # Copy to target row (overwrite existing value)
            # If source_value is None, we'll set it to empty string for consistency
            if source_value is None:
                row[col_name] = ""
            else:
                row[col_name] = source_value


def insert_column_task(work: Dict[str, List[Dict[str, Any]]],
                       sheet_name: str,
                       column_name: str,
                       logger: Optional[Any] = None) -> Optional[str]:
    """
    Ensure that a given column exists on a given sheet.
    
    If the column doesn't exist, it is created and set to "" (empty string) for every row.
    If it does exist, existing values are left as-is, but missing row keys are filled in as empty strings.
    
    This task does NOT:
    - Delete columns
    - Move column positions
    - Change any values except to add "" where the column didn't exist
    
    Example:
        Insert column - Contacts output | GPT_Notes
        - Ensures that "GPT_Notes" exists on "Contacts output"
        - If it doesn't exist, creates it and sets "" for every row
        - If it already exists, leaves values as-is and backfills missing keys as ""
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        sheet_name: Name of sheet in work
        column_name: Name of column to ensure exists
        logger: Optional logger for debug logging
        
    Returns:
        Error message string if an error occurred, None if successful
    """
    # Step 1: Fetch sheet
    rows = work.get(sheet_name)
    
    if rows is None:
        error_msg = f"Sheet '{sheet_name}' not found"
        if logger:
            logger.debug(f"insert_column_task: {error_msg}")
        return error_msg
    
    try:
        # Step 2: Check if column exists
        # Determine whether column_name exists in the sheet
        # Check if any row has that key
        column_exists = False
        if len(rows) > 0:
            column_exists = any(column_name in row for row in rows)
        
        # Step 3: If column does NOT exist, create it with empty strings for all rows
        if not column_exists:
            for row in rows:
                row[column_name] = ""
        else:
            # Step 4: If column DOES exist, ensure every row has a key for this column
            # Backfill missing keys with empty strings
            for row in rows:
                if column_name not in row:
                    row[column_name] = ""
        
        # No other changes - we don't alter other columns, reorder rows, or perform any I/O
        return None
    except Exception as e:
        # Step 6: Error handling - catch unexpected exceptions
        error_msg = f"Unexpected error in insert_column_task: {str(e)}"
        if logger:
            logger.debug(f"insert_column_task: {error_msg}")
        return error_msg


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
    Execute an AI recipe task with optional conditional execution.
    
    This function:
    1. Reads input sheet data
    2. For each row, evaluates condition if present (skips row if condition is false)
    3. Builds prompts for eligible rows by substituting {ColumnName} placeholders
    4. Calls AI API with batching, concurrency limits, and retries
    5. Writes results to output column in output sheet
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        task: RecipeTask with type TASK_AI and params containing:
            - input_sheet_name: Name of input sheet
            - output_sheet_name: Name of output sheet
            - output_column_name: Name of column to write results to
            - model_name: Model name (will be mapped to OpenRouter identifier)
            - prompt_template: Prompt template with {ColumnName} placeholders
            - condition: Optional dict with 'column_name' and 'substring' for conditional execution
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
    condition = task.params.get("condition")  # Optional condition
    
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
    
    # Helper function to evaluate condition for a row
    def evaluate_condition(row: Dict[str, Any]) -> bool:
        """
        Evaluate the condition for a row.
        
        Returns:
            True if condition is None (unconditional) or if condition evaluates to True.
            False if condition evaluates to False.
        """
        if condition is None:
            return True  # Unconditional: process all rows
        
        # Get condition column value
        condition_column_name = condition["column_name"]
        condition_substring = condition["substring"]
        
        # Get value from row (stringified, trimmed)
        row_value = row.get(condition_column_name)
        if row_value is None:
            row_value = ""
        else:
            row_value = str(row_value).strip()
        
        # Case-insensitive contains check
        return condition_substring.lower() in row_value.lower()
    
    # Build prompts for each row
    async def process_row(row_index: int, row: Dict[str, Any]) -> None:
        """Process a single row: check condition, build prompt, call AI, write result."""
        try:
            # Evaluate condition first - if false, skip this row entirely
            if not evaluate_condition(row):
                # Condition is false: skip this row, leave output cell unchanged
                return
            
            # Condition is true (or no condition): proceed with AI processing
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
    # Note: Rows that don't pass the condition will be skipped in process_row
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
    task: RecipeTask,
    exa_client: Exa,
    semaphore: asyncio.Semaphore
) -> None:
    """
    Execute an Exa recipe task with optional conditional execution.
    
    This function:
    1. Reads input sheet data
    2. For each row, evaluates condition if present (skips row if condition is false)
    3. Checks if website is non-empty and output column is empty (idempotent)
    4. Calls Exa API with batching, concurrency limits, and retries
    5. Writes results to output column
    
    Args:
        work: Dictionary mapping sheet names to their row lists (mutated)
        task: RecipeTask with type TASK_EXA and params containing:
            - sheet_name: Name of sheet to process
            - website_column: Name of column containing website URLs
            - output_column: Name of column to write results to
            - condition: Optional dict with 'column_name' and 'type' for conditional execution
        exa_client: Exa client instance
        semaphore: Semaphore to limit concurrent Exa requests
        
    Raises:
        ValueError: If sheet is not found or other configuration errors
    """
    sheet_name = task.params["sheet_name"]
    website_column = task.params["website_column"]
    output_column = task.params["output_column"]
    condition = task.params.get("condition")  # Optional condition
    
    # Resolve sheet
    rows = work.get(sheet_name)
    if rows is None:
        raise ValueError(f"Sheet '{sheet_name}' not found in work dictionary")
    
    # Determine column headers from first row (if available)
    column_headers = set()
    if len(rows) > 0:
        column_headers = set(rows[0].keys())
    
    # Validate that website_column exists
    if len(rows) > 0 and website_column not in column_headers:
        raise ValueError(f"Website column '{website_column}' not found in sheet '{sheet_name}'")
    
    # Ensure output_column exists in all rows
    for row in rows:
        if output_column not in row:
            row[output_column] = ""
    
    # Helper function to evaluate condition for a row
    def evaluate_condition(row: Dict[str, Any]) -> bool:
        """
        Evaluate the condition for a row.
        
        Returns:
            True if condition is None (unconditional) or if condition evaluates to True.
            False if condition evaluates to False.
        """
        if condition is None:
            return True  # Unconditional: process all rows
        
        # Get condition column value
        condition_column_name = condition["column_name"]
        condition_type = condition["type"]
        
        # Get value from row (stringified, trimmed)
        row_value = row.get(condition_column_name)
        if row_value is None:
            row_value = ""
        else:
            row_value = str(row_value).strip()
        
        # Evaluate based on condition type
        if condition_type == "is_not_empty":
            return len(row_value) > 0
        
        # Unknown condition type
        return False
    
    # Process each row
    async def process_row(row_index: int, row: Dict[str, Any]) -> None:
        """Process a single row: check condition, check website/output, call Exa, write result."""
        try:
            # Evaluate condition first - if false, skip this row entirely
            if not evaluate_condition(row):
                # Condition is false: skip this row, leave output cell unchanged
                return
            
            # Get website value
            website = row.get(website_column)
            if website is None:
                website = ""
            else:
                website = str(website).strip()
            
            # If website is empty, skip this row
            if not website:
                return
            
            # Check if output column already has a non-empty value (idempotent behavior)
            current_output = row.get(output_column)
            if current_output is not None:
                current_output_str = str(current_output).strip()
                if current_output_str:
                    # Output already populated, skip to avoid re-calling Exa
                    return
            
            # Call Exa with semaphore for concurrency control
            async with semaphore:
                exa_summary = await _call_exa_with_retry(exa_client, website)
            
            # Write result to output column
            # If Exa call failed, leave empty (or optionally set error marker)
            if exa_summary is None:
                # Leave empty on failure (per spec: don't overwrite with error markers)
                row[output_column] = ""
            else:
                row[output_column] = exa_summary
                
        except Exception as e:
            # Per-row error: leave output empty and continue processing other rows
            # This ensures one row failure doesn't crash the entire task
            row[output_column] = ""
            # Note: We don't log here since we don't have a logger in this context
            # The caller can handle logging if needed
    
    # Process all rows concurrently (with semaphore limiting concurrency)
    # Batch processing: process in chunks to avoid overwhelming the system
    batch_size = 50
    for batch_start in range(0, len(rows), batch_size):
        batch_end = min(batch_start + batch_size, len(rows))
        batch_rows = rows[batch_start:batch_end]
        
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
    print(f"[RECIPE][RUN] initial work sheets: {list(work.keys())}")
    
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
            elif task.type == TASK_REMOVE_TEXT:
                remove_text_task(
                    work,
                    task.params["sheet_name"],
                    task.params["column_name"],
                    task.params["phrases"]
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
            elif task.type == TASK_COPY_BY_KEY:
                copy_by_key_task(
                    work,
                    task.params["source_sheet_name"],
                    task.params["source_key_column"],
                    task.params["target_sheet_name"],
                    task.params["target_key_column"]
                )
            elif task.type == TASK_INSERT_COLUMN:
                try:
                    error_msg = insert_column_task(
                        work,
                        task.params["sheet_name"],
                        task.params["column_name"]
                    )
                    if error_msg:
                        # Record task-level error but don't crash the recipe
                        errors.append(f"Row {task.row_index}: {error_msg}")
                        continue
                except Exception as e:
                    # Catch any unexpected exceptions and handle gracefully
                    errors.append(f"Row {task.row_index}: Insert column task failed: {str(e)}")
                    continue
            elif task.type == TASK_AI:
                # AI tasks are async, so we need to run them in an event loop
                # Since run_recipe is sync, we use asyncio.run()
                # Note: asyncio.run() creates a new event loop, so this is safe even if
                # called from a sync context (which is the case in worker.py)
                try:
                    asyncio.run(run_ai_task(work, task, ai_client, ai_semaphore))
                except Exception as e:
                    raise ValueError(f"AI task failed on row {task.row_index}: {str(e)}")
            elif task.type == TASK_EXA:
                # Exa tasks are async, so we need to run them in an event loop
                # Since run_recipe is sync, we use asyncio.run()
                # Note: asyncio.run() creates a new event loop, so this is safe even if
                # called from a sync context (which is the case in worker.py)
                try:
                    asyncio.run(run_exa_task(work, task, exa_client, exa_semaphore))
                except Exception as e:
                    raise ValueError(f"Exa task failed on row {task.row_index}: {str(e)}")
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
    
    # Check if there were any runtime errors (e.g., from insert_column_task)
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

