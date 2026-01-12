"""
Automated cold email workflow - Website enrichment script.

This script processes websites from Supabase, enriches them with Exa.AI summaries,
and categorizes them using OpenRouter LLM models.

Production-ready version with batch processing for large datasets.
"""

import asyncio
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client, Client
from exa_py import Exa
from openai import AsyncOpenAI
import sheet_export

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SEMAPHORE_LIMIT = 5  # Max concurrent requests (respects Exa's 10 QPS limit)
BATCH_SIZE = 50  # Number of rows to fetch and process per batch
RUN_CLEANING = False  # Enable/disable full-table cleaning phase

# Exa Configuration
RUN_EXA = True  # Enable/disable Exa fetching (if False, Exa is never called)

# Default model configuration (used for all steps if prompts table doesn't specify)
DEFAULT_MODEL_PRIMARY = "google/gemini-2.5-flash"
DEFAULT_MODEL_FALLBACK = "openai/gpt-5-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 250


class EnrichmentWorkflow:
    """Handles the enrichment workflow for prospect websites."""
    
    def __init__(self, project_id: str):
        """
        Initialize global clients for Supabase, Exa.AI, and OpenRouter (reused across all batches).
        
        Args:
            project_id: Project ID to scope all operations (required)
        """
        if not project_id:
            raise ValueError("project_id is required and cannot be empty")
        
        required_vars = [SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY]
        if RUN_EXA:
            required_vars.append(EXA_API_KEY)
        
        if not all(required_vars):
            missing = []
            if not SUPABASE_URL:
                missing.append("SUPABASE_URL")
            if not SUPABASE_KEY:
                missing.append("SUPABASE_KEY")
            if not OPENROUTER_API_KEY:
                missing.append("OPENROUTER_API_KEY")
            if RUN_EXA and not EXA_API_KEY:
                missing.append("EXA_API_KEY")
            
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please check your .env file."
            )
        
        self.project_id = project_id
        
        # Initialize global Supabase client (reused across all batches)
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Initialize global Exa.AI client (reused across all batches) - only if RUN_EXA is True
        self.exa = Exa(api_key=EXA_API_KEY) if RUN_EXA else None
        
        # Initialize global OpenRouter client (reused across all batches)
        # OpenRouter is OpenAI-compatible, so we use AsyncOpenAI with OpenRouter base URL
        self.openrouter_client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "ProspectV2Workflow"
            }
        )
        
        # Initialize observability counters
        self.prospects_seen = 0
        self.prospects_enriched = 0
        self.prospects_deferred_db_write = 0
        self.ai_calls_attempted = 0
        self.ai_calls_retried = 0
        self.db_writes_attempted = 0
        self.db_write_retries = 0
        self.db_write_failures = 0
        self.exa_failures = 0
        self.prompts_loaded_count = 0
        self.prospects_skipped_summary_too_short = 0
        self.steps_skipped_missing_placeholders = 0
        self.prospects_skipped_missing_placeholders = 0
        self.prospects_blocked_missing_placeholders = 0
        self.steps_skipped_missing_dependencies = 0
        self.steps_skipped_run_if_false = 0
        self.prospects_blocked_malformed_run_if = 0
        self.rows_claimed = 0
        self.rows_skipped_already_claimed = 0
        self.exa_api_calls_made = 0
        self.exa_api_skipped_because_existing_exa_summary = 0
        self.exa_api_skipped_because_run_exa_false = 0
        # Branch observability counters
        self.branch_steps_seen = 0
        self.branch_steps_matched = 0
        self.branch_steps_else_taken = 0
        self.branch_steps_no_match_skipped = 0
        self.branch_parse_errors = 0
        self.branch_dependency_skips = 0
        
        # Prompts loaded from database (list of dicts with step_order, prompt_text, run_if)
        self.prompts: List[Dict[str, Any]] = []
        
        # Run ID for idempotency within this run
        self.run_id = None
        
        # Database run ID from runs table (for checking if run is still active)
        self.db_run_id = None
        
        # Run token for supersession detection (set from first fetched row)
        self.run_token = None
        
        # Flag to track if run was superseded
        self.run_superseded = False
    
    def is_run_active(self, run_id: Optional[str] = None) -> bool:
        """
        Check if a run is still active (status='running').
        
        Args:
            run_id: Run ID to check. If None, uses self.db_run_id.
            
        Returns:
            True if run status is 'running', False otherwise.
        """
        # Use provided run_id or fall back to self.db_run_id
        check_run_id = run_id or self.db_run_id
        
        # If no run_id available, assume active (backward compatibility)
        if not check_run_id:
            return True
        
        try:
            resp = (
                self.supabase.table("runs")
                .select("status")
                .eq("id", check_run_id)
                .single()
                .execute()
            )
            if not resp.data:
                return False
            return resp.data.get("status") == "running"
        except Exception:
            return False
    
    def _normalize_website(self, website: str) -> str:
        """
        Normalize website URL: remove protocol, www, trailing slashes, convert to lowercase.
        
        Args:
            website: Website URL to normalize
            
        Returns:
            Normalized website string
        """
        if not website:
            return ""
        
        # Convert to lowercase
        normalized = website.lower().strip()
        
        # Remove protocol (case insensitive)
        for protocol in ['https://', 'http://']:
            if normalized.startswith(protocol):
                normalized = normalized[len(protocol):]
        
        # Remove www. (case insensitive)
        if normalized.startswith('www.'):
            normalized = normalized[4:]
        
        # Remove trailing slash
        normalized = normalized.rstrip('/')
        
        return normalized
    
    def _is_junk_website(self, website: str) -> bool:
        """
        Check if website is junk (contains error phrases or is empty).
        
        Args:
            website: Website string to check
            
        Returns:
            True if website is junk, False otherwise
        """
        if not website or not website.strip():
            return True
        
        website_lower = website.lower()
        junk_phrases = ['not found', 'unreachable']
        
        return any(phrase in website_lower for phrase in junk_phrases)
    
    def fetch_prompts(self) -> List[Dict[str, Any]]:
        """
        Fetch active prompts from Supabase prompts table, ordered by step_order.
        
        Returns:
            List of prompt dictionaries with step_order, prompt_text, run_if, and branch in execution order
        """
        try:
            # Try to select branch column, but handle gracefully if it doesn't exist
            try:
                response = (
                    self.supabase.table('prompts')
                    .select('step_order, prompt_text, run_if, branch')
                    .eq('is_active', True)
                    .eq('project_id', self.project_id)
                    .order('step_order', desc=False)
                    .execute()
                )
            except Exception as e:
                # If branch column doesn't exist, fall back to selecting without it
                error_str = str(e).lower()
                if 'column' in error_str and ('does not exist' in error_str or 'not found' in error_str):
                    print("  [WARNING] 'branch' column not found in prompts table. Continuing without branch support.")
                    response = (
                        self.supabase.table('prompts')
                        .select('step_order, prompt_text, run_if')
                        .eq('is_active', True)
                        .eq('project_id', self.project_id)
                        .order('step_order', desc=False)
                        .execute()
                    )
                else:
                    raise
            
            if not response.data:
                return []
            
            # Extract prompts as dicts with step_order, prompt_text, run_if, and branch
            prompts = []
            for row in response.data:
                prompts.append({
                    'step_order': row.get('step_order'),
                    'prompt_text': row.get('prompt_text', ''),
                    'run_if': row.get('run_if') or '',  # Convert None to empty string
                    'branch': row.get('branch') or ''  # Convert None to empty string
                })
            
            self.prompts_loaded_count = len(prompts)
            return prompts
            
        except Exception as e:
            print(f"Error fetching prompts from Supabase: {str(e)}")
            return []
    
    async def clean_data(self) -> None:
        """
        Clean prospect data: normalize URLs, remove junk, deduplicate.
        
        Performs:
        1. Normalization: Clean website URLs (remove protocol, www, trailing slashes, lowercase)
        2. Remove Junk: Delete rows with junk websites
        3. Deduplication: Keep oldest row per cleaned URL, delete duplicates
        """
        print("\n[0/4] Starting data cleaning phase...")
        print("-" * 60)
        
        # Statistics
        cleaned_count = 0
        junk_deleted_count = 0
        duplicates_deleted_count = 0
        
        # Fetch all prospects with pagination
        print("Fetching all prospects for cleaning...")
        all_prospects = []
        offset = 0
        fetch_batch_size = 1000
        
        while True:
            try:
                response = (
                    self.supabase.table('prospects')
                    .select('id, website')
                    .eq('project_id', self.project_id)
                    .range(offset, offset + fetch_batch_size - 1)
                    .execute()
                )
            
                if not response.data:
                    break
                
                all_prospects.extend(response.data)
                
                if len(response.data) < fetch_batch_size:
                    break
                
                offset += fetch_batch_size
                print(f"  Fetched {len(all_prospects)} prospects so far...")
            except Exception as e:
                print(f"  Error fetching prospects: {str(e)}")
                break
        
        if not all_prospects:
            print("No prospects found to clean.")
            return
        
        print(f"  Total prospects fetched: {len(all_prospects)}")
        print("-" * 60)
        
        # Phase 1: Normalization and Junk Removal
        print("\nPhase 1: Normalizing URLs and removing junk...")
        updates_batch = []
        delete_ids = []
        
        for prospect in all_prospects:
            prospect_id = prospect.get('id')
            website = prospect.get('website', '')
            
            # Check for junk
            if self._is_junk_website(website):
                delete_ids.append(prospect_id)
                junk_deleted_count += 1
                continue
            
            # Normalize website
            normalized = self._normalize_website(website)
            
            # Only update if website changed
            if normalized != website:
                updates_batch.append({
                    'id': prospect_id,
                    'website': normalized
                })
                cleaned_count += 1
        
        # Batch update normalized websites
        if updates_batch:
            print(f"  Updating {len(updates_batch)} normalized URLs...")
            # Process updates in smaller batches to avoid payload limits
            update_batch_size = 100
            for i in range(0, len(updates_batch), update_batch_size):
                batch = updates_batch[i:i + update_batch_size]
                try:
                    # Update each row individually (Supabase doesn't support bulk update easily)
                    for item in batch:
                        def update_item(item_id=item['id'], website=item['website']):
                            return (
                                self.supabase.table('prospects')
                                .update({'website': website})
                                .eq('id', item_id)
                                .execute()
                            )
                        await asyncio.to_thread(update_item)
                except Exception as e:
                    print(f"  Error updating batch: {str(e)}")
        
        # Batch delete junk rows
        if delete_ids:
            print(f"  Deleting {len(delete_ids)} junk rows...")
            delete_batch_size = 100
            for i in range(0, len(delete_ids), delete_batch_size):
                batch_ids = delete_ids[i:i + delete_batch_size]
                try:
                    # Delete in batches using .in_() filter
                    def delete_batch(ids=batch_ids):
                        return (
                            self.supabase.table('prospects')
                            .delete()
                            .in_('id', ids)
                            .execute()
                        )
                    await asyncio.to_thread(delete_batch)
                except Exception as e:
                    print(f"  Error deleting junk batch: {str(e)}")
        
        print(f"  ✓ Normalized {cleaned_count} URLs")
        print(f"  ✓ Deleted {junk_deleted_count} junk rows")
        
        # Phase 2: Deduplication
        print("\nPhase 2: Removing duplicates...")
        
        # Re-fetch all remaining prospects to get normalized URLs
        print("  Re-fetching prospects for deduplication...")
        remaining_prospects = []
        offset = 0
        
        while True:
            try:
                response = (
                    self.supabase.table('prospects')
                    .select('id, website')
                    .eq('project_id', self.project_id)
                    .order('id', desc=False)  # Order by id ascending (oldest first)
                    .range(offset, offset + fetch_batch_size - 1)
                    .execute()
                )
            
                if not response.data:
                    break
                
                remaining_prospects.extend(response.data)
                
                if len(response.data) < fetch_batch_size:
                    break
                
                offset += fetch_batch_size
            except Exception as e:
                print(f"  Error fetching prospects: {str(e)}")
                break
        
        # Group by normalized website
        website_groups = {}
        for prospect in remaining_prospects:
            prospect_id = prospect.get('id')
            website = prospect.get('website', '')
            normalized = self._normalize_website(website)
            
            if normalized:  # Skip empty websites
                if normalized not in website_groups:
                    website_groups[normalized] = []
                website_groups[normalized].append(prospect_id)
        
        # Find duplicates and mark for deletion (keep oldest by id)
        duplicate_ids_to_delete = []
        for normalized_url, ids in website_groups.items():
            if len(ids) > 1:
                # Sort by id (ascending), keep first (oldest), delete rest
                sorted_ids = sorted(ids)
                duplicate_ids_to_delete.extend(sorted_ids[1:])
        
        # Delete duplicates in batches
        if duplicate_ids_to_delete:
            print(f"  Found {len(duplicate_ids_to_delete)} duplicate rows to delete...")
            delete_batch_size = 100
            for i in range(0, len(duplicate_ids_to_delete), delete_batch_size):
                batch_ids = duplicate_ids_to_delete[i:i + delete_batch_size]
                try:
                    def delete_duplicates(ids=batch_ids):
                        return (
                            self.supabase.table('prospects')
                            .delete()
                            .in_('id', ids)
                            .execute()
                        )
                    await asyncio.to_thread(delete_duplicates)
                except Exception as e:
                    print(f"  Error deleting duplicate batch: {str(e)}")
            
            duplicates_deleted_count = len(duplicate_ids_to_delete)
        
        print(f"  ✓ Removed {duplicates_deleted_count} duplicates")
        
        # Print summary
        print("-" * 60)
        print(f"\n✓ Data cleaning complete!")
        print(f"  Cleaned {cleaned_count} URLs. Deleted {junk_deleted_count} junk rows. Removed {duplicates_deleted_count} duplicates.")
        print("-" * 60)
    
    async def test_supabase_connection(self) -> bool:
        """
        Test the Supabase connection by attempting to query the prospects table.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to fetch a single row to test the connection
            response = (
                self.supabase.table('prospects')
                .select('id')
                .eq('project_id', self.project_id)
                .limit(1)
                .execute()
            )
            print("✓ Successfully connected to Supabase!")
            print(f"✓ Verified access to 'prospects' table")
            return True
        except Exception as e:
            print(f"✗ Error connecting to Supabase: {str(e)}")
            print(f"✗ Please verify your SUPABASE_URL and SUPABASE_KEY are correct")
            return False
    
    async def get_exa_summary(self, website: str) -> Optional[str]:
        """
        Get company summary from Exa.AI for a given website.
        
        Args:
            website: The website URL to summarize
            
        Returns:
            Summary string or None if error occurs
        """
        if not RUN_EXA or not self.exa:
            return None
        
        try:
            # Ensure website has protocol
            if not website.startswith(('http://', 'https://')):
                website = f"https://{website}"
            
            # Call Exa.AI to get summary (using global client)
            response = self.exa.search_and_contents(
                query=f"company information about {website}",
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
                elif hasattr(result, 'url'):
                    # If no text, try to get content from URL
                    content_response = self.exa.get_contents(
                        ids=[result.id],
                        text={"max_characters": 1000}
                    )
                    if content_response.results and len(content_response.results) > 0:
                        return content_response.results[0].text
            
            return None
        except Exception as e:
            print(f"Error getting Exa summary for {website}: {str(e)}")
            return None
    
    def _extract_placeholders(self, prompt_template: str) -> List[str]:
        """
        Extract placeholder names from a prompt template.
        
        Args:
            prompt_template: The prompt template string
            
        Returns:
            List of placeholder names (without braces)
        """
        placeholders = re.findall(r'\{(\w+)\}', prompt_template)
        return placeholders
    
    def _parse_run_if(self, run_if: str) -> Optional[Dict[str, str]]:
        """
        Parse a run_if condition string.
        
        Grammar: {placeholder} contains: "value"
        - Case-insensitive
        - Spacing-insensitive
        - Only operator supported: contains:
        - Value must be in double quotes
        
        Args:
            run_if: The run_if condition string (may be empty/None)
            
        Returns:
            Dict with keys 'placeholder' and 'value' if valid, None if blank, raises ValueError if malformed
        """
        if not run_if or not run_if.strip():
            return None  # Blank run_if means always run
        
        # Normalize: remove extra whitespace
        normalized = ' '.join(run_if.split())
        
        # Pattern: {placeholder} contains: "value"
        # Case-insensitive matching
        pattern = r'\{(\w+)\}\s+contains:\s+"([^"]+)"'
        match = re.search(pattern, normalized, re.IGNORECASE)
        
        if not match:
            raise ValueError(f"Malformed run_if: {run_if}")
        
        return {
            'placeholder': match.group(1),
            'value': match.group(2)
        }
    
    def _evaluate_run_if(self, run_if_parsed: Optional[Dict[str, str]], variables: Dict[str, str]) -> bool:
        """
        Evaluate a parsed run_if condition against variables.
        
        Args:
            run_if_parsed: Parsed run_if dict (None means always run)
            variables: Dictionary of variable names to values
            
        Returns:
            True if condition passes (or run_if is blank), False otherwise
        """
        if run_if_parsed is None:
            return True  # Blank run_if means always run
        
        placeholder = run_if_parsed['placeholder']
        expected_value = run_if_parsed['value']
        
        # Get the actual value from variables (default to empty string)
        actual_value = variables.get(placeholder, '')
        
        # Case-insensitive comparison
        return expected_value.lower() in actual_value.lower()
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace: trim leading/trailing, normalize internal runs to single spaces.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        # Strip leading/trailing whitespace
        normalized = text.strip()
        # Normalize internal whitespace runs to single spaces
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _normalize_branch_prompt_text(self, prompt_text: str) -> str:
        """
        Normalize Branch DSL prompt text by removing quotes and handling empty strings.
        
        This handles cases like:
        - '""' -> '' (quoted empty string)
        - "''" -> '' (quoted empty string with single quotes)
        - '  ""  ' -> '' (quoted empty with whitespace)
        - '"text"' -> 'text' (remove one layer of quotes)
        - "'text'" -> 'text' (remove one layer of single quotes)
        
        Args:
            prompt_text: The prompt text to normalize
            
        Returns:
            Normalized prompt text (empty string if input is empty/whitespace/quoted-empty)
        """
        if not prompt_text:
            return ""
        
        # Strip outer whitespace first
        normalized = prompt_text.strip()
        
        # If empty after strip, return empty string
        if not normalized:
            return ""
        
        # Check if wrapped in matching quotes (single or double) and remove one layer
        # Handle double quotes: "text" or ""
        if len(normalized) >= 2 and normalized[0] == '"' and normalized[-1] == '"':
            normalized = normalized[1:-1]
        # Handle single quotes: 'text' or ''
        elif len(normalized) >= 2 and normalized[0] == "'" and normalized[-1] == "'":
            normalized = normalized[1:-1]
        
        # After unquoting, strip again and check if empty/whitespace
        normalized = normalized.strip()
        
        # If result is empty, whitespace-only, or exactly "" / '' (quoted-empty), normalize to ""
        if not normalized or normalized == '""' or normalized == "''":
            return ""
        
        return normalized
    
    def _parse_branch(self, branch_text: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Parse a Branch DSL string into placeholder and rules.
        
        Format:
        BRANCH({placeholder}):
        "match1" :: prompt text 1
        "match2" :: prompt text 2
        ELSE :: prompt text else
        
        Args:
            branch_text: The branch DSL string
            
        Returns:
            Tuple of (branch_placeholder, rules_list)
            rules_list contains dicts with keys: 'type' ('match' or 'else'), 'match_value' (for match type), 'prompt_text'
            
        Raises:
            ValueError: If branch_text is malformed
        """
        if not branch_text or not branch_text.strip():
            raise ValueError("Branch text is empty")
        
        lines = branch_text.strip().split('\n')
        rules = []
        branch_placeholder = None
        
        # Find and parse header line: BRANCH({placeholder}):
        header_found = False
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Check for BRANCH header
            if line.upper().startswith('BRANCH('):
                # Extract placeholder from BRANCH({placeholder}):
                match = re.search(r'BRANCH\(\{(\w+)\}\s*\)\s*:', line, re.IGNORECASE)
                if not match:
                    raise ValueError(f"Malformed BRANCH header: {line}. Expected format: BRANCH({{placeholder}}):")
                branch_placeholder = match.group(1)
                header_found = True
                break
        
        if not header_found:
            raise ValueError("Missing BRANCH header. Expected format: BRANCH({placeholder}):")
        
        if not branch_placeholder:
            raise ValueError("Could not extract placeholder from BRANCH header")
        
        # Parse rule lines (after header)
        header_index = -1
        for i, line in enumerate(lines):
            if line.strip().upper().startswith('BRANCH('):
                header_index = i
                break
        
        if header_index == -1:
            raise ValueError("Could not find BRANCH header line")
        
        # Process lines after header
        for line in lines[header_index + 1:]:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Check for ELSE rule (case-insensitive)
            if line.upper().startswith('ELSE'):
                # Parse ELSE :: prompt_text
                else_match = re.match(r'ELSE\s*::\s*(.*)', line, re.IGNORECASE | re.DOTALL)
                if not else_match:
                    raise ValueError(f"Malformed ELSE rule: {line}. Expected format: ELSE :: prompt_text")
                prompt_text = else_match.group(1).strip()
                # Normalize prompt text (remove quotes, handle empty strings)
                prompt_text = self._normalize_branch_prompt_text(prompt_text)
                rules.append({
                    'type': 'else',
                    'prompt_text': prompt_text
                })
            else:
                # Parse match rule: "match" :: prompt_text
                match_rule = re.match(r'"([^"]+)"\s*::\s*(.*)', line, re.DOTALL)
                if not match_rule:
                    raise ValueError(f"Malformed match rule: {line}. Expected format: \"match\" :: prompt_text")
                match_value = match_rule.group(1)
                prompt_text = match_rule.group(2).strip()
                # Normalize prompt text (remove quotes, handle empty strings)
                prompt_text = self._normalize_branch_prompt_text(prompt_text)
                rules.append({
                    'type': 'match',
                    'match_value': match_value,
                    'prompt_text': prompt_text
                })
        
        if not rules:
            raise ValueError("No rules found in branch (must have at least one match rule or ELSE rule)")
        
        return branch_placeholder, rules
    
    def _select_branch_prompt(self, parsed_branch: tuple[str, List[Dict[str, Any]]], variables: Dict[str, str]) -> tuple[Optional[str], Optional[str]]:
        """
        Select a prompt from parsed branch rules based on variable values.
        
        Args:
            parsed_branch: Tuple of (branch_placeholder, rules_list) from _parse_branch
            variables: Dictionary of variable names to values
            
        Returns:
            Tuple of (selected_prompt_text, rule_type) where rule_type is 'match', 'else', or None
            Returns (None, None) if no match and no ELSE
        """
        branch_placeholder, rules = parsed_branch
        
        # Get the branch variable value
        branch_value = variables.get(branch_placeholder, '')
        
        # Normalize branch value: trim and normalize whitespace
        normalized_branch_value = self._normalize_whitespace(branch_value).lower()
        
        # Try to match rules in order (first match wins)
        for rule in rules:
            if rule['type'] == 'match':
                # Normalize match value
                normalized_match = self._normalize_whitespace(rule['match_value']).lower()
                
                # Case-insensitive contains match
                if normalized_match in normalized_branch_value:
                    return (rule['prompt_text'], 'match')
            # else rules are handled after all match rules
        
        # No match found - check for ELSE rule
        for rule in rules:
            if rule['type'] == 'else':
                return (rule['prompt_text'], 'else')
        
        # No match and no ELSE
        return (None, None)
    
    def _is_blank(self, value: Any) -> bool:
        """
        Check if a value is missing/blank.
        
        A placeholder value counts as missing if:
        - It is None OR
        - After converting to string and stripping whitespace, it is empty.
        
        Args:
            value: The value to check
            
        Returns:
            True if value is missing/blank, False otherwise
        """
        if value is None:
            return True
        if not isinstance(value, str):
            value = str(value)
        return not value.strip()
    
    def _is_exa_summary_empty(self, exa_summary: Optional[str]) -> bool:
        """
        Check if exa_summary is empty.
        
        Empty means: NULL or "" or whitespace-only.
        
        Args:
            exa_summary: The exa_summary value to check
            
        Returns:
            True if empty, False otherwise
        """
        if exa_summary is None:
            return True
        if not isinstance(exa_summary, str):
            return True
        return not exa_summary.strip()
    
    def _build_company_summary(self, exa_summary: Optional[str], short_description: Optional[str]) -> str:
        """
        Build company_summary from exa_summary and short_description.
        
        Args:
            exa_summary: Exa summary (may be None or empty)
            short_description: Short description (may be None or empty)
            
        Returns:
            Combined company_summary string
        """
        exa_part = (exa_summary or "").strip()
        desc_part = (short_description or "").strip()
        
        if RUN_EXA and exa_part and desc_part:
            return f"{exa_part} / {desc_part}"
        elif RUN_EXA and exa_part:
            return exa_part
        elif desc_part:
            return desc_part
        else:
            return ""
    
    def _format_prompt(self, prompt_template: str, variables: Dict[str, str]) -> str:
        """
        Format a prompt template with available variables, safely handling missing placeholders.
        
        Args:
            prompt_template: The prompt template string
            variables: Dictionary of variable names to values
            
        Returns:
            Formatted prompt string
        """
        # Build a dict with all possible placeholders, defaulting to empty string
        safe_vars = {
            'company_summary': variables.get('company_summary', ''),
            'exa_summary': variables.get('exa_summary', ''),
            'short_description': variables.get('short_description', ''),
            'step1_output': variables.get('step1_output', ''),
            'step2_output': variables.get('step2_output', ''),
            'step3_output': variables.get('step3_output', ''),
            'step4_output': variables.get('step4_output', ''),
            'step5_output': variables.get('step5_output', ''),
        }
        
        # Check for any placeholders in the template that we don't support
        import re
        placeholders = re.findall(r'\{(\w+)\}', prompt_template)
        for placeholder in placeholders:
            if placeholder not in safe_vars:
                print(f"  [WARNING] Prompt references unknown placeholder: {{{placeholder}}}")
                # Add it as empty string to avoid KeyError
                safe_vars[placeholder] = ''
        
        # Use safe formatting - if a placeholder is missing, treat as empty string
        try:
            return prompt_template.format(**safe_vars)
        except (KeyError, ValueError) as e:
            print(f"  [WARNING] Error formatting prompt: {str(e)}")
            # Fallback: replace placeholders manually
            formatted = prompt_template
            for key, value in safe_vars.items():
                formatted = formatted.replace(f'{{{key}}}', str(value))
            return formatted
    
    async def _call_llm_with_retry(
        self,
        step_num: int,
        prompt: str,
        model_primary: str,
        model_fallback: str,
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """
        Call OpenRouter LLM with retry logic (primary model with fallback).
        
        Args:
            step_num: Step number (for logging)
            prompt: The prompt to send to the LLM
            model_primary: Primary model to use
            model_fallback: Fallback model if primary fails
            temperature: Temperature setting
            max_tokens: Max tokens setting
            
        Returns:
            Response text or None if all attempts fail
        """
        max_retries = 3
        backoff_delays = [1, 2]  # 1s, then 2s
        
        # Try primary model first
        for attempt in range(max_retries):
            self.ai_calls_attempted += 1
            try:
                response = await self.openrouter_client.chat.completions.create(
                    model=model_primary,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if response and response.choices and len(response.choices) > 0:
                    result = response.choices[0].message.content.strip()
                    return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.ai_calls_retried += 1
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    print(f"  [Step {step_num}] Primary model attempt {attempt + 1} failed, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"  [Step {step_num}] Primary model failed after {max_retries} attempts, trying fallback...")
        
        # Try fallback model
        for attempt in range(max_retries):
            self.ai_calls_attempted += 1
            try:
                response = await self.openrouter_client.chat.completions.create(
                    model=model_fallback,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if response and response.choices and len(response.choices) > 0:
                    result = response.choices[0].message.content.strip()
                    print(f"  [Step {step_num}] Fallback model succeeded")
                    return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.ai_calls_retried += 1
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    print(f"  [Step {step_num}] Fallback model attempt {attempt + 1} failed, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"  [Step {step_num}] Fallback model failed after {max_retries} attempts")
        
        return None
    
    async def _run_enrichment_step(
        self,
        step_num: int,
        prompt_template: str,
        variables: Dict[str, str]
    ) -> Optional[str]:
        """
        Run a single enrichment step.
        
        Args:
            step_num: Step number (1-based)
            prompt_template: The prompt template string
            variables: Dictionary of variables for prompt formatting
            
        Returns:
            Step output or None if failed
        """
        # Format prompt with available variables
        prompt = self._format_prompt(prompt_template, variables)
        
        # Use default model configuration
        model_primary = DEFAULT_MODEL_PRIMARY
        model_fallback = DEFAULT_MODEL_FALLBACK
        temperature = DEFAULT_TEMPERATURE
        max_tokens = DEFAULT_MAX_TOKENS
        
        # Call LLM with retry
        return await self._call_llm_with_retry(
            step_num=step_num,
            prompt=prompt,
            model_primary=model_primary,
            model_fallback=model_fallback,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def _verify_write(
        self,
        prospect_id: Any,
        expected_data: Dict[str, Any]
    ) -> bool:
        """
        Verify that a write to Supabase was successful by re-reading the row.
        
        Args:
            prospect_id: The ID of the prospect to verify
            expected_data: Dictionary of expected field values
            
        Returns:
            True if verification passes, False otherwise
        """
        try:
            def read_prospect():
                return (
                    self.supabase.table('prospects')
                    .select('id, status, exa_summary, company_summary, step1_output, step2_output, step3_output, step4_output, step5_output')
                    .eq('id', prospect_id)
                    .eq('project_id', self.project_id)
                    .execute()
                )
            
            response = await asyncio.to_thread(read_prospect)
            
            if not response.data or len(response.data) == 0:
                return False
            
            row = response.data[0]
            
            # Verify status
            if 'status' in expected_data:
                if row.get('status') != expected_data['status']:
                    return False
            
            # Verify exa_summary
            if 'exa_summary' in expected_data and expected_data['exa_summary'] is not None:
                if row.get('exa_summary') != expected_data['exa_summary']:
                    return False
            
            # Verify company_summary
            if 'company_summary' in expected_data and expected_data['company_summary'] is not None:
                if row.get('company_summary') != expected_data['company_summary']:
                    return False
            
            # Verify step outputs (check up to step5)
            for step_num in range(1, 6):
                step_key = f'step{step_num}_output'
                if step_key in expected_data:
                    if row.get(step_key) != expected_data[step_key]:
                        return False
            
            return True
        except Exception as e:
            print(f"  [VERIFY] Error verifying write for prospect {prospect_id}: {str(e)}")
            return False
    
    async def _write_prospect_with_retry(
        self,
        prospect_id: Any,
        exa_summary: Optional[str],
        company_summary: str,
        step_outputs: Dict[int, str],
        status: str,
        db_write_semaphore: asyncio.Semaphore,
        blocked_reason: Optional[str] = None
    ) -> bool:
        """
        Write prospect data to Supabase with retry logic and verification.
        
        Args:
            prospect_id: The ID of the prospect to update
            exa_summary: The summary from Exa.AI (can be None)
            company_summary: The computed company_summary
            step_outputs: Dictionary of step outputs (key: step_num, value: output)
            status: Status to set ('enriched', 'skipped', or 'blocked')
            db_write_semaphore: Semaphore to serialize DB writes (concurrency=1)
            blocked_reason: Optional reason for blocking (only used if status='blocked')
            
        Returns:
            True if write succeeded and was verified, False otherwise
        """
        async with db_write_semaphore:
            self.db_writes_attempted += 1
            
            # Build update data
            update_data = {
                'status': status,
                'company_summary': company_summary
            }
            
            if exa_summary is not None:
                update_data['exa_summary'] = exa_summary
            
            # Include step outputs (support up to step5)
            # Write all step outputs (including empty strings for skipped steps)
            for step_num in range(1, 6):
                step_key = f'step{step_num}_output'
                if step_num in step_outputs:
                    # Write the value even if it's empty string (for skipped steps)
                    update_data[step_key] = step_outputs[step_num]
            
            # Optionally include blocked_reason if provided (column may not exist)
            if blocked_reason is not None:
                update_data['blocked_reason'] = blocked_reason
            
            # Clear claim fields when row finishes successfully (enriched/skipped/blocked)
            # This releases the row so it's available for future processing if needed
            if status in ('enriched', 'skipped', 'blocked'):
                update_data['processing_run_id'] = None
                update_data['processing_started_at'] = None
            
            # Retry logic with linear backoff (1s, 2s, 3s)
            max_retries = 3
            backoff_delays = [1, 2, 3]
            
            for attempt in range(max_retries):
                if attempt > 0:
                    self.db_write_retries += 1
                    delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                    print(f"  [DB WRITE] Prospect {prospect_id} (run_id: {self.run_id}): Retry {attempt} after {delay}s...")
                    await asyncio.sleep(delay)
                
                try:
                    # Perform the write
                    def update_prospect():
                        return (
                            self.supabase.table('prospects')
                            .update(update_data)
                            .eq('id', prospect_id)
                            .eq('project_id', self.project_id)
                            .execute()
                        )
                    
                    await asyncio.to_thread(update_prospect)
                    
                    # Verify the write
                    if await self._verify_write(prospect_id, update_data):
                        # Success - immediately discard cached data (memory safety)
                        return True
                    else:
                        print(f"  [DB WRITE] Prospect {prospect_id} (run_id: {self.run_id}): Write verification failed on attempt {attempt + 1}")
                        if attempt == max_retries - 1:
                            self.db_write_failures += 1
                            return False
                
                except Exception as e:
                    error_str = str(e).lower()
                    # If blocked_reason column doesn't exist, retry without it
                    if blocked_reason is not None and ('column' in error_str or 'field' in error_str or 'does not exist' in error_str):
                        if attempt == 0:
                            # Remove blocked_reason and retry
                            update_data_without_reason = {k: v for k, v in update_data.items() if k != 'blocked_reason'}
                            update_data = update_data_without_reason
                            print(f"  [DB WRITE] Prospect {prospect_id} (run_id: {self.run_id}): blocked_reason column not found, retrying without it...")
                            continue
                    
                    print(f"  [DB WRITE] Prospect {prospect_id} (run_id: {self.run_id}): Write attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        self.db_write_failures += 1
                        return False
            
            return False
    
    async def enrich_prospect(
        self,
        prospect: Dict[str, Any],
        ai_semaphore: asyncio.Semaphore,
        db_write_semaphore: asyncio.Semaphore,
        failed_this_run: set
    ) -> None:
        """
        Enrich a single prospect with Exa summary and multi-step OpenRouter enrichment.
        Caches all AI results in memory and writes to Supabase only once after all steps complete.
        
        Args:
            prospect: Dictionary containing prospect data (must have 'id' and 'website' fields)
            ai_semaphore: Semaphore to limit concurrent AI requests
            db_write_semaphore: Semaphore to serialize DB writes (concurrency=1)
            failed_this_run: Set of normalized websites that have failed in this run
        """
        self.prospects_seen += 1
        prospect_id = prospect.get('id')
        website = prospect.get('website')
        
        if not website:
            print(f"  [ERROR] Prospect {prospect_id} (run_id: {self.run_id}) has no website")
            return
        
        # Check if this prospect's normalized website has already failed in this run
        normalized_website = self._normalize_website(website)
        if normalized_website in failed_this_run:
            print(f"  [SKIP] Prospect {prospect_id} (normalized: {normalized_website}, run_id: {self.run_id}): Already failed in this run, skipping")
            return
        
        # Get existing data
        existing_exa_summary = prospect.get('exa_summary')
        short_description = prospect.get('short_description')
        
        # Get existing step outputs
        existing_step_outputs = {}
        for step in range(1, 6):
            step_key = f'step{step}_output'
            if step_key in prospect and prospect[step_key]:
                existing_step_outputs[step] = prospect[step_key]
        
        # Check if prospect is eligible (has at least one blank required step)
        num_prompts = len(self.prompts)
        has_blank_step = False
        for step_num in range(1, num_prompts + 1):
            if step_num not in existing_step_outputs or not existing_step_outputs[step_num]:
                has_blank_step = True
                break
        
        if not has_blank_step:
            # All required steps already have output, skip this prospect
            return
        
        async with ai_semaphore:
            # In-memory cache for AI results
            cached_exa_summary: Optional[str] = None
            cached_company_summary: str = ""
            cached_step_outputs: Dict[int, str] = {}
            
            try:
                # Copy existing step outputs
                cached_step_outputs = existing_step_outputs.copy()
                
                # Check if run is still active before starting enrichment
                if not self.is_run_active():
                    print(f"Run {self.db_run_id} no longer active. Aborting mid-run.")
                    return
                
                # STEP 1: Get Exa summary (conditional on RUN_EXA and exa_summary emptiness)
                if not RUN_EXA:
                    # RUN_EXA is False: NEVER call Exa API
                    self.exa_api_skipped_because_run_exa_false += 1
                    cached_exa_summary = None
                elif not self._is_exa_summary_empty(existing_exa_summary):
                    # RUN_EXA is True but exa_summary already has text (including from CSV): reuse it
                    cached_exa_summary = existing_exa_summary
                    self.exa_api_skipped_because_existing_exa_summary += 1
                else:
                    # RUN_EXA is True AND exa_summary is empty: call Exa API
                    # Check if run is still active before calling Exa
                    if not self.is_run_active():
                        print(f"Run {self.db_run_id} no longer active. Aborting mid-run.")
                        return
                    
                    self.exa_api_calls_made += 1
                    cached_exa_summary = await self.get_exa_summary(website)
                    
                    # Check if run is still active after Exa call
                    if not self.is_run_active():
                        print(f"Run {self.db_run_id} no longer active. Aborting mid-run.")
                        return
                    
                    # If Exa fails, accept it and proceed (do not mark as failed)
                    if not cached_exa_summary or not cached_exa_summary.strip():
                        print(f"  [EXA] No summary found for {website} (ID: {prospect_id}, run_id: {self.run_id}) - proceeding without Exa")
                        self.exa_failures += 1
                        cached_exa_summary = None
                
                # STEP 2: Build company_summary
                cached_company_summary = self._build_company_summary(cached_exa_summary, short_description)
                
                # STEP 3: Check company_summary length gate
                if not cached_company_summary or len(cached_company_summary) <= 50:
                    # Skip this prospect - company_summary too short
                    print(f"  [SKIP] Prospect {prospect_id} (run_id: {self.run_id}): company_summary too short (length: {len(cached_company_summary)})")
                    self.prospects_skipped_summary_too_short += 1
                    
                    # Write skipped status with company_summary and exa_summary
                    write_success = await self._write_prospect_with_retry(
                        prospect_id,
                        cached_exa_summary,
                        cached_company_summary,
                        {},
                        'skipped',
                        db_write_semaphore
                    )
                    
                    if not write_success:
                        print(f"  [ERROR] Failed to write skipped status for prospect {prospect_id}")
                        failed_this_run.add(normalized_website)
                    
                    return
                
                # STEP 4: Run all required enrichment steps sequentially, caching outputs in memory
                num_prompts = len(self.prompts)
                
                for step_num in range(1, num_prompts + 1):
                    # Check if run is still active before processing next step
                    if not self.is_run_active():
                        print(f"Run {self.db_run_id} no longer active. Aborting mid-run.")
                        return
                    
                    # Skip if this step already has output
                    if step_num in cached_step_outputs and cached_step_outputs[step_num]:
                        continue
                    
                    # Build variables dict for prompt formatting
                    variables = {
                        'company_summary': cached_company_summary,
                        'exa_summary': cached_exa_summary or '',
                        'short_description': short_description or '',
                    }
                    
                    # Add prior step outputs
                    for prev_step in range(1, step_num):
                        variables[f'step{prev_step}_output'] = cached_step_outputs.get(prev_step, '')
                    
                    # Get prompt dict for this step
                    prompt_dict = self.prompts[step_num - 1]  # 0-indexed
                    prompt_template = prompt_dict['prompt_text']
                    run_if = prompt_dict.get('run_if', '') or ''
                    branch = prompt_dict.get('branch', '') or ''
                    
                    # Parse and evaluate run_if condition (OUTER GATE - evaluated before Branch)
                    try:
                        run_if_parsed = self._parse_run_if(run_if)
                        run_if_passes = self._evaluate_run_if(run_if_parsed, variables)
                    except ValueError as e:
                        # Malformed run_if - block the prospect
                        malformed_reason = f"Malformed run_if: {run_if}"
                        print(f"  [BLOCKED] Prospect {prospect_id} (run_id: {self.run_id}) website={normalized_website} step={step_num} - {malformed_reason}")
                        self.prospects_blocked_malformed_run_if += 1
                        
                        # Persist blocked status to DB immediately
                        write_success = await self._write_prospect_with_retry(
                            prospect_id,
                            cached_exa_summary,
                            cached_company_summary,
                            {},  # Empty dict - do not overwrite existing step outputs
                            'blocked',
                            db_write_semaphore,
                            blocked_reason=malformed_reason
                        )
                        
                        if not write_success:
                            print(f"  [ERROR] Failed to write blocked status for prospect {prospect_id}")
                        
                        # Exit processing for this prospect (no further steps attempted)
                        return
                    
                    if not run_if_passes:
                        # run_if evaluated to False - skip this step (Branch is NOT evaluated)
                        print(f"  [SKIP RUN_IF] Prospect {prospect_id} (run_id: {self.run_id}) step={step_num} run_if condition not met")
                        cached_step_outputs[step_num] = ""  # Set to empty string
                        self.steps_skipped_run_if_false += 1
                        continue  # Continue to next step
                    
                    # Run If passed - now handle Branch selection (if present)
                    selected_prompt_template = prompt_template  # Default to original prompt_text
                    
                    if branch:
                        # Branch is present - parse and select prompt
                        self.branch_steps_seen += 1
                        try:
                            parsed_branch = self._parse_branch(branch)
                            selected_prompt, rule_type = self._select_branch_prompt(parsed_branch, variables)
                            
                            if selected_prompt is None:
                                # No match and no ELSE - skip step
                                print(f"  [SKIP BRANCH] Prospect {prospect_id} (run_id: {self.run_id}) step={step_num} no branch match and no ELSE")
                                cached_step_outputs[step_num] = ""  # Set to empty string
                                self.branch_steps_no_match_skipped += 1
                                continue  # Continue to next step
                            
                            # Normalize selected prompt: handle None, then strip whitespace
                            normalized_selected_prompt = selected_prompt.strip() if selected_prompt else ""
                            
                            if normalized_selected_prompt == "":
                                # Selected prompt is empty/whitespace - skip step (DO NOT call LLM)
                                print(f"  [SKIP BRANCH] Prospect {prospect_id} (run_id: {self.run_id}) step={step_num} branch selected empty/whitespace prompt")
                                cached_step_outputs[step_num] = ""  # Set to empty string
                                # Increment appropriate counters
                                if rule_type == 'match':
                                    self.branch_steps_matched += 1
                                elif rule_type == 'else':
                                    self.branch_steps_else_taken += 1
                                continue  # Continue to next step (skip dependency checks and LLM call)
                            
                            # Track which rule type was selected
                            if rule_type == 'match':
                                self.branch_steps_matched += 1
                            elif rule_type == 'else':
                                self.branch_steps_else_taken += 1
                            
                            # Use selected prompt from branch
                            selected_prompt_template = selected_prompt
                            
                        except ValueError as e:
                            # Malformed branch - block the prospect
                            branch_error_msg = str(e)
                            malformed_reason = f"Branch parse error at step {step_num}: {branch_error_msg}"
                            print(f"  [BLOCKED] Prospect {prospect_id} (run_id: {self.run_id}) website={normalized_website} step={step_num} - {malformed_reason}")
                            self.branch_parse_errors += 1
                            
                            # Persist blocked status to DB immediately
                            write_success = await self._write_prospect_with_retry(
                                prospect_id,
                                cached_exa_summary,
                                cached_company_summary,
                                {},  # Empty dict - do not overwrite existing step outputs
                                'blocked',
                                db_write_semaphore,
                                blocked_reason=malformed_reason
                            )
                            
                            if not write_success:
                                print(f"  [ERROR] Failed to write blocked status for prospect {prospect_id}")
                            
                            # Exit processing for this prospect (no further steps attempted)
                            return
                    
                    # Check if selected prompt (branch or non-branch) is empty/whitespace - skip if so
                    normalized_prompt = selected_prompt_template.strip() if selected_prompt_template else ""
                    if normalized_prompt == "":
                        # Prompt is empty/whitespace - skip step (DO NOT call LLM)
                        print(f"  [SKIP EMPTY PROMPT] Prospect {prospect_id} (run_id: {self.run_id}) step={step_num} prompt is empty/whitespace")
                        cached_step_outputs[step_num] = ""  # Set to empty string
                        continue  # Continue to next step (skip dependency checks and LLM call)
                    
                    # Now check dependencies on the selected prompt (after branch selection)
                    placeholders = self._extract_placeholders(selected_prompt_template)
                    has_empty_dependency = False
                    empty_dependencies = []
                    
                    for placeholder in placeholders:
                        placeholder_value = variables.get(placeholder, '')
                        # Treat None/blank as empty string
                        if not placeholder_value or not str(placeholder_value).strip():
                            has_empty_dependency = True
                            empty_dependencies.append(placeholder)
                    
                    if has_empty_dependency:
                        # Skip this step - set output to empty string and continue
                        print(f"  [SKIP DEPENDENCY] Prospect {prospect_id} (run_id: {self.run_id}) step={step_num} missing dependencies: {','.join(empty_dependencies)}")
                        cached_step_outputs[step_num] = ""  # Set to empty string
                        self.steps_skipped_missing_dependencies += 1
                        if branch:
                            self.branch_dependency_skips += 1
                        continue  # Continue to next step
                    
                    # Check if run is still active before calling LLM
                    if not self.is_run_active():
                        print(f"Run {self.db_run_id} no longer active. Aborting mid-run.")
                        return
                    
                    # All checks passed - run the step with selected prompt
                    step_result = await self._run_enrichment_step(step_num, selected_prompt_template, variables)
                    
                    # Check if run is still active after LLM call
                    if not self.is_run_active():
                        print(f"Run {self.db_run_id} no longer active. Aborting mid-run.")
                        return
                    
                    if not step_result:
                        # Step failed after all retries - this is a final failure
                        print(f"  [ERROR] Prospect {prospect_id} (normalized: {normalized_website}, run_id: {self.run_id}): Step {step_num} failed after all retries")
                        failed_this_run.add(normalized_website)
                        return
                    
                    # Store the result in memory cache
                    cached_step_outputs[step_num] = step_result
                
                # All steps processed (some may have been skipped) - perform single final write to Supabase
                write_success = await self._write_prospect_with_retry(
                    prospect_id,
                    cached_exa_summary,
                    cached_company_summary,
                    cached_step_outputs,
                    'enriched',
                    db_write_semaphore
                )
                
                if write_success:
                    self.prospects_enriched += 1
                    # Immediately discard cached data (memory safety)
                    cached_exa_summary = None
                    cached_company_summary = ""
                    cached_step_outputs = {}
                else:
                    # DB write failed after all retries - this is a final failure
                    self.prospects_deferred_db_write += 1
                    print(f"  [DEFERRED] Prospect {prospect_id} (normalized: {normalized_website}, run_id: {self.run_id}): All retries failed. Row remains status='new' for next run.")
                    failed_this_run.add(normalized_website)
                    # Discard cached data even on failure (memory safety)
                    cached_exa_summary = None
                    cached_company_summary = ""
                    cached_step_outputs = {}
                
            except Exception as e:
                # Catch any unexpected errors - treat as final failure
                print(f"  [ERROR] Unexpected error processing prospect {prospect_id} ({website}, normalized: {normalized_website}, run_id: {self.run_id}): {str(e)}")
                failed_this_run.add(normalized_website)
                # Discard cached data on error (memory safety)
                cached_exa_summary = None
                cached_company_summary = ""
                cached_step_outputs = {}
    
    def _claim_rows_atomically(self, row_ids: List[Any]) -> List[Any]:
        """
        Atomically claim rows by setting processing_run_id and processing_started_at.
        Only claims rows that are still eligible (status='new' AND project_id matches AND run_token matches AND (processing_run_id IS NULL OR processing_started_at < now() - interval '30 minutes')).
        Uses UPDATE ... RETURNING id to ensure 100% atomic and provably safe claiming.
        
        Args:
            row_ids: List of row IDs to attempt to claim
            
        Returns:
            List of row IDs that were successfully claimed (only IDs returned from UPDATE ... RETURNING)
        """
        if not row_ids or not self.run_id or not self.run_token:
            return []
        
        print("[CLAIM] Using update().execute() without select() (compat mode)")
        
        claimed_ids = []
        thirty_min_ago = (datetime.utcnow() - timedelta(minutes=30)).isoformat()
        update_data = {
            'processing_run_id': self.run_id,
            'processing_started_at': datetime.utcnow().isoformat()
        }
        
        # Per-batch counters for diagnostics
        attempted_count = len(row_ids)
        claimed_count = 0
        no_match_count = 0
        error_count = 0
        exception_count = 0
        
        # Claim each row atomically by updating with eligibility check in WHERE clause
        # Use UPDATE ... RETURNING id to get only rows that were actually claimed
        for row_id in row_ids:
            row_claimed = False
            
            # Attempt A: Update rows where status='new' AND processing_run_id IS NULL AND project_id matches AND run_token matches
            where_a = f"id={row_id} AND status='new' AND processing_run_id IS NULL AND project_id='{self.project_id}' AND run_token='{self.run_token}'"
            try:
                response = (
                    self.supabase.table('prospects')
                    .update(update_data)
                    .eq('id', row_id)
                    .eq('status', 'new')
                    .eq('project_id', self.project_id)
                    .eq('run_token', self.run_token)
                    .is_('processing_run_id', 'null')
                    .execute()
                )
                
                # Extract IDs from response.data
                extracted_ids = []
                if response.data:
                    if isinstance(response.data, list):
                        # List of dicts
                        for row in response.data:
                            if isinstance(row, dict) and 'id' in row:
                                extracted_ids.append(row['id'])
                    elif isinstance(response.data, dict):
                        # Single dict
                        if 'id' in response.data:
                            extracted_ids.append(response.data['id'])
                
                response_data_len = len(response.data) if response.data else 0
                extracted_count = len(extracted_ids)
                
                # Check for REAL error signals:
                # - status_code >= 400
                # - error is present/non-empty
                # - code is present/non-empty
                # - message is present/non-empty
                has_real_error = False
                error_info_parts = []
                
                if hasattr(response, 'status_code') and response.status_code is not None:
                    status_code = response.status_code
                    if status_code >= 400:
                        has_real_error = True
                        error_info_parts.append(f'status_code={status_code}')
                    # Note: status_code in 200-299 is NOT an error by itself
                
                if hasattr(response, 'error') and response.error:
                    has_real_error = True
                    error_info_parts.append(f'error="{response.error}"')
                
                if hasattr(response, 'code') and response.code:
                    has_real_error = True
                    error_info_parts.append(f'code={response.code}')
                
                if hasattr(response, 'message') and response.message:
                    has_real_error = True
                    error_info_parts.append(f'message="{response.message}"')
                
                error_info = ' '.join(error_info_parts) if error_info_parts else None
                
                # Log diagnostic - SUCCESS only if we extracted at least one ID
                if extracted_count > 0:
                    # Successfully claimed
                    for returned_id in extracted_ids:
                        claimed_ids.append(returned_id)
                        self.rows_claimed += 1
                        claimed_count += 1
                        row_claimed = True
                        break
                elif has_real_error:
                    # REAL error in response
                    error_count += 1
                    print(f"  [CLAIM ERROR] row_id={row_id} attempt=A where=\"{where_a}\" returned_rows={response_data_len} extracted_ids={extracted_count} {error_info}")
                else:
                    # No match (row didn't match WHERE clause) - normal NO-MATCH case
                    no_match_count += 1
                    print(f"  [CLAIM NO-MATCH] row_id={row_id} attempt=A where=\"{where_a}\" returned_rows={response_data_len} extracted_ids={extracted_count}")
                    
            except Exception as e:
                exception_count += 1
                exc_type = type(e).__name__
                exc_msg = str(e)
                print(f"  [CLAIM EXCEPTION] row_id={row_id} attempt=A where=\"{where_a}\" exception_type={exc_type} exception_msg=\"{exc_msg}\"")
            
            # If Attempt A succeeded, skip Attempt B
            if row_claimed:
                continue
            
            # Attempt B: Update rows where status='new' AND processing_started_at < thirty_min_ago AND project_id matches AND run_token matches
            where_b = f"id={row_id} AND status='new' AND processing_started_at<'{thirty_min_ago}' AND project_id='{self.project_id}' AND run_token='{self.run_token}'"
            try:
                response = (
                    self.supabase.table('prospects')
                    .update(update_data)
                    .eq('id', row_id)
                    .eq('status', 'new')
                    .eq('project_id', self.project_id)
                    .eq('run_token', self.run_token)
                    .lt('processing_started_at', thirty_min_ago)
                    .execute()
                )
                
                # Extract IDs from response.data
                extracted_ids = []
                if response.data:
                    if isinstance(response.data, list):
                        # List of dicts
                        for row in response.data:
                            if isinstance(row, dict) and 'id' in row:
                                extracted_ids.append(row['id'])
                    elif isinstance(response.data, dict):
                        # Single dict
                        if 'id' in response.data:
                            extracted_ids.append(response.data['id'])
                
                response_data_len = len(response.data) if response.data else 0
                extracted_count = len(extracted_ids)
                
                # Check for REAL error signals:
                # - status_code >= 400
                # - error is present/non-empty
                # - code is present/non-empty
                # - message is present/non-empty
                has_real_error = False
                error_info_parts = []
                
                if hasattr(response, 'status_code') and response.status_code is not None:
                    status_code = response.status_code
                    if status_code >= 400:
                        has_real_error = True
                        error_info_parts.append(f'status_code={status_code}')
                    # Note: status_code in 200-299 is NOT an error by itself
                
                if hasattr(response, 'error') and response.error:
                    has_real_error = True
                    error_info_parts.append(f'error="{response.error}"')
                
                if hasattr(response, 'code') and response.code:
                    has_real_error = True
                    error_info_parts.append(f'code={response.code}')
                
                if hasattr(response, 'message') and response.message:
                    has_real_error = True
                    error_info_parts.append(f'message="{response.message}"')
                
                error_info = ' '.join(error_info_parts) if error_info_parts else None
                
                # Log diagnostic - SUCCESS only if we extracted at least one ID
                if extracted_count > 0:
                    # Successfully claimed
                    for returned_id in extracted_ids:
                        claimed_ids.append(returned_id)
                        self.rows_claimed += 1
                        claimed_count += 1
                        row_claimed = True
                        break
                elif has_real_error:
                    # REAL error in response
                    error_count += 1
                    print(f"  [CLAIM ERROR] row_id={row_id} attempt=B where=\"{where_b}\" returned_rows={response_data_len} extracted_ids={extracted_count} {error_info}")
                else:
                    # No match (row didn't match WHERE clause) - normal NO-MATCH case
                    no_match_count += 1
                    print(f"  [CLAIM NO-MATCH] row_id={row_id} attempt=B where=\"{where_b}\" returned_rows={response_data_len} extracted_ids={extracted_count}")
                    
            except Exception as e:
                exception_count += 1
                exc_type = type(e).__name__
                exc_msg = str(e)
                print(f"  [CLAIM EXCEPTION] row_id={row_id} attempt=B where=\"{where_b}\" exception_type={exc_type} exception_msg=\"{exc_msg}\"")
            
            # If both attempts failed, update global counter
            if not row_claimed:
                self.rows_skipped_already_claimed += 1
        
        # Print per-batch summary
        print(f"  [CLAIM SUMMARY] attempted={attempted_count} claimed={claimed_count} no_match={no_match_count} error={error_count} exception={exception_count}")
        
        return claimed_ids
    
    def fetch_batch(self, limit: int = BATCH_SIZE) -> list:
        """
        Fetch a batch of prospects with status 'new' from Supabase and atomically claim them.
        Only fetches rows that are eligible: status='new' AND project_id matches AND run_token matches AND (processing_run_id IS NULL OR processing_started_at < now() - interval '30 minutes').
        Also fetch existing step outputs to support "only fill blanks" logic.
        
        Sets run_token from first row if not already set.
        Detects superseded runs (different run_token) and exits gracefully.
        
        Returns ONLY rows whose IDs were returned from UPDATE ... RETURNING id in _claim_rows_atomically().
        No inference of success - relies exclusively on returned IDs.
        
        Args:
            limit: Number of rows to fetch
            
        Returns:
            List of prospect dictionaries (only rows that were successfully claimed via UPDATE ... RETURNING)
            Returns empty list if run is superseded
        """
        if not self.run_id:
            print("Error: run_id not set. Cannot claim rows.")
            return []
        
        try:
            # Calculate 30 minutes ago for eligibility check
            thirty_min_ago = (datetime.utcnow() - timedelta(minutes=30)).isoformat()
            
            # Fetch eligible rows: status='new' AND project_id matches AND (processing_run_id IS NULL OR processing_started_at < thirty_min_ago)
            # We'll fetch a larger batch to account for rows that might not be claimable
            # Then we'll claim them atomically and return only the claimed ones
            
            # First, fetch rows with status='new' for this project_id that are potentially eligible
            # We'll fetch more than limit to account for rows that might be claimed by other runs
            fetch_limit = limit * 2  # Fetch 2x to account for potential conflicts
            
            response = (
                self.supabase.table('prospects')
                .select('id, website, short_description, exa_summary, company_summary, step1_output, step2_output, step3_output, step4_output, step5_output, processing_run_id, processing_started_at, run_token')
                .eq('status', 'new')
                .eq('project_id', self.project_id)
                .limit(fetch_limit)
                .execute()
            )
            
            if not response.data:
                return []
            
            # Set run_token from first row if not already set
            if self.run_token is None:
                first_row = response.data[0]
                row_run_token = first_row.get('run_token')
                if row_run_token:
                    self.run_token = row_run_token
                    print(f"Using run_token={self.run_token} from first fetched row")
                else:
                    print("WARNING: First row has no run_token. This may indicate schema not ready.")
            
            # Filter to only eligible rows with matching run_token
            eligible_rows = []
            for row in response.data:
                row_run_token = row.get('run_token')
                processing_run_id = row.get('processing_run_id')
                processing_started_at = row.get('processing_started_at')
                
                # Check if run is superseded (different run_token)
                if row_run_token and self.run_token and row_run_token != self.run_token:
                    print(f"Run superseded — exiting. project_id={self.project_id} run_token={self.run_token} (found run_token={row_run_token})")
                    self.run_superseded = True
                    return []  # Return empty list to exit gracefully
                
                # Check eligibility: matching run_token AND (NULL processing_run_id OR old processing_started_at)
                is_eligible = (
                    (not row_run_token or row_run_token == self.run_token) and
                    (processing_run_id is None or
                     (processing_started_at and processing_started_at < thirty_min_ago))
                )
                
                if is_eligible:
                    eligible_rows.append(row)
            
            # Limit to requested batch size
            eligible_rows = eligible_rows[:limit]
            
            if not eligible_rows:
                return []
            
            # Extract row IDs for claiming
            row_ids = [row['id'] for row in eligible_rows]
            
            # Atomically claim the rows using UPDATE ... RETURNING id
            # This returns ONLY the IDs that were actually claimed in the UPDATE
            claimed_ids = self._claim_rows_atomically(row_ids)
            
            # Return only the rows whose IDs were returned from UPDATE ... RETURNING
            # No inference - rely exclusively on returned IDs
            claimed_rows = [row for row in eligible_rows if row['id'] in claimed_ids]
            
            return claimed_rows
            
        except Exception as e:
            print(f"Error fetching prospects batch: {str(e)}")
            return []
    
    async def run(self, run_id: str) -> None:
        """
        Main workflow execution with batch processing.
        
        Fetches 50 rows at a time, processes them, then fetches the next batch.
        Continues until no more rows with status='new' are found.
        
        Args:
            run_id: Run ID from the runs table (for checking if run is still active)
        """
        # Store database run_id for active checks
        self.db_run_id = run_id
        
        # Generate run_id for idempotency within this run
        self.run_id = str(uuid.uuid4())[:8]
        
        # Capture start time at the very beginning
        start_time = time.time()
        
        print("=" * 60)
        print("Starting enrichment workflow (Production Mode)")
        print(f"Using project_id={self.project_id}")
        print(f"Run ID: {self.run_id}")
        print("=" * 60)
        
        # Test Supabase connection first
        print("\n[1/5] Testing Supabase connection...")
        connection_ok = await self.test_supabase_connection()
        if not connection_ok:
            print("Cannot proceed without a valid Supabase connection.")
            return
        
        # Fetch prompts from database
        print("\n[2/5] Fetching prompts from database...")
        self.prompts = self.fetch_prompts()
        if not self.prompts:
            print("ERROR: No active prompts found in database. Cannot proceed.")
            print("Please ensure the 'prompts' table has at least one row with is_active=true.")
            return
        
        print(f"  ✓ Loaded {len(self.prompts)} active prompts")
        for i, prompt_dict in enumerate(self.prompts, 1):
            prompt_text = prompt_dict['prompt_text']
            run_if = prompt_dict.get('run_if', '') or ''
            run_if_display = f" [run_if: {run_if}]" if run_if else ""
            prompt_display = prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
            print(f"    Step {i}: {prompt_display}{run_if_display}")
        
        print("\n[3/5] Initializing clients...")
        print(f"  ✓ Supabase client initialized")
        if RUN_EXA:
            print(f"  ✓ Exa.AI client initialized")
        else:
            print(f"  ⚠ Exa.AI disabled (RUN_EXA=False)")
        print(f"  ✓ OpenRouter client initialized")
        print(f"  ✓ AI Semaphore limit: {SEMAPHORE_LIMIT} concurrent requests")
        print(f"  ✓ DB Write Semaphore: 1 (serialized writes)")
        print(f"  ✓ Batch size: {BATCH_SIZE} rows per batch")
        print(f"  ✓ Number of enrichment steps: {len(self.prompts)}")
        
        # Data cleaning phase
        print("\n[4/5] Data cleaning phase...")
        if RUN_CLEANING:
            print("Cleaning enabled: running clean_data()")
            await self.clean_data()
        else:
            print("Cleaning disabled: skipping clean_data()")
        
        print("\n[5/5] Starting batch processing...")
        print("-" * 60)
        
        # Create semaphores
        # AI semaphore for rate limiting (shared across all batches)
        ai_semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
        # DB write semaphore with concurrency=1 (serialized writes)
        db_write_semaphore = asyncio.Semaphore(1)
        
        # Initialize failed_this_run set to track normalized websites that have failed in this run
        failed_this_run = set()
        
        # Batch processing loop
        total_processed = 0
        batch_number = 0
        
        while True:
            batch_number += 1
            
            # Check if run is still active before processing next batch
            if not self.is_run_active():
                print(f"Run {self.db_run_id} no longer active. Aborting mid-run.")
                return
            
            # Fetch next batch of 50 rows
            print(f"\n[Batch {batch_number}] Fetching {BATCH_SIZE} prospects with status='new'...")
            batch = self.fetch_batch(BATCH_SIZE)
            
            # If no rows returned, check if we were superseded
            if not batch:
                if self.run_superseded:
                    print(f"[Batch {batch_number}] Run superseded — exiting gracefully. project_id={self.project_id} run_token={self.run_token}")
                else:
                    print(f"[Batch {batch_number}] No more prospects to process. Job complete!")
                break
            
            # Filter out prospects whose normalized website is already in failed_this_run
            filtered_batch = []
            skipped_count = 0
            for prospect in batch:
                website = prospect.get('website', '')
                normalized_website = self._normalize_website(website)
                if normalized_website in failed_this_run:
                    skipped_count += 1
                    prospect_id = prospect.get('id')
                    print(f"  [SKIP] Prospect {prospect_id} (normalized: {normalized_website}, run_id: {self.run_id}): Already failed in this run, skipping")
                else:
                    filtered_batch.append(prospect)
            
            if skipped_count > 0:
                print(f"[Batch {batch_number}] Skipped {skipped_count} prospects already failed in this run")
            
            if not filtered_batch:
                print(f"[Batch {batch_number}] All prospects in batch were skipped. Fetching next batch...")
                continue
            
            print(f"[Batch {batch_number}] Processing {len(filtered_batch)} prospects (skipped {skipped_count})...")
            
            # Process this batch concurrently with AI semaphore
            tasks = [
                self.enrich_prospect(prospect, ai_semaphore, db_write_semaphore, failed_this_run)
                for prospect in filtered_batch
            ]
            
            # Wait for all tasks in this batch to complete
            await asyncio.gather(*tasks)
            
            total_processed += len(filtered_batch)
            print(f"[Batch {batch_number}] Batch Complete: {len(filtered_batch)} rows processed (Total: {total_processed})")
            
            # Check if run is still active after processing batch
            if not self.is_run_active():
                print(f"Run {self.db_run_id} no longer active. Aborting mid-run.")
                self.run_superseded = True
                break
        
        # Export results to Google Sheets if run completed successfully (not superseded)
        if not self.run_superseded:
            # Check if sheet_id exists before attempting export
            print("\n[EXPORT] Checking for sheet_id...")
            try:
                # Fetch sheet_id from prompts table
                sheet_id_response = (
                    self.supabase.table('prompts')
                    .select('sheet_id')
                    .eq('project_id', self.project_id)
                    .eq('is_active', True)
                    .limit(1)
                    .execute()
                )
                
                sheet_id = None
                if sheet_id_response.data and len(sheet_id_response.data) > 0:
                    sheet_id = sheet_id_response.data[0].get('sheet_id')
                
                if not sheet_id or not sheet_id.strip():
                    print(f"  [EXPORT] No sheet_id for project {self.project_id}, skipping export")
                else:
                    print("\n[EXPORT] Exporting results to Google Sheets...")
                    try:
                        sheet_export.export_results_to_google_sheets(self.project_id)
                    except Exception as e:
                        # Log error but don't crash - export failures are non-fatal
                        print(f"  [EXPORT] Error during export (non-fatal): {str(e)}")
            except Exception as e:
                # Log error but don't crash - export failures are non-fatal
                print(f"  [EXPORT] Error checking sheet_id (non-fatal): {str(e)}")
        else:
            print("\n[EXPORT] Run was superseded, skipping Google Sheets export")
        
        # Capture end time at the very end
        end_time = time.time()
        
        # Calculate duration
        duration_seconds = end_time - start_time
        duration_minutes = int(duration_seconds // 60)
        duration_secs = int(duration_seconds % 60)
        
        # Calculate average speed
        if total_processed > 0:
            avg_seconds_per_prospect = duration_seconds / total_processed
        else:
            avg_seconds_per_prospect = 0
        
        print("-" * 60)
        print(f"\n✓ Enrichment workflow completed!")
        print(f"✓ Run ID: {self.run_id}")
        print(f"✓ Total prospects processed: {total_processed}")
        print(f"✓ Total time elapsed: {duration_minutes} min {duration_secs} sec")
        print(f"✓ Average speed: {avg_seconds_per_prospect:.2f} seconds per prospect")
        print("-" * 60)
        
        # Print observability counters
        print("\n=== OBSERVABILITY METRICS ===")
        print(f"prospects_seen: {self.prospects_seen}")
        print(f"prospects_enriched: {self.prospects_enriched}")
        print(f"prospects_deferred_db_write: {self.prospects_deferred_db_write}")
        print(f"prospects_skipped_summary_too_short: {self.prospects_skipped_summary_too_short}")
        print(f"prospects_blocked_missing_placeholders: {self.prospects_blocked_missing_placeholders}")
        print(f"prospects_blocked_malformed_run_if: {self.prospects_blocked_malformed_run_if}")
        print(f"steps_skipped_missing_placeholders: {self.steps_skipped_missing_placeholders}")
        print(f"steps_skipped_missing_dependencies: {self.steps_skipped_missing_dependencies}")
        print(f"steps_skipped_run_if_false: {self.steps_skipped_run_if_false}")
        print(f"prospects_skipped_missing_placeholders: {self.prospects_skipped_missing_placeholders}")
        print(f"prompts_loaded_count: {self.prompts_loaded_count}")
        print(f"exa_failures: {self.exa_failures}")
        print(f"exa_api_calls_made: {self.exa_api_calls_made}")
        print(f"exa_api_skipped_because_existing_exa_summary: {self.exa_api_skipped_because_existing_exa_summary}")
        print(f"exa_api_skipped_because_run_exa_false: {self.exa_api_skipped_because_run_exa_false}")
        print(f"ai_calls_attempted: {self.ai_calls_attempted}")
        print(f"ai_calls_retried: {self.ai_calls_retried}")
        print(f"db_writes_attempted: {self.db_writes_attempted}")
        print(f"db_write_retries: {self.db_write_retries}")
        print(f"db_write_failures: {self.db_write_failures}")
        print(f"rows_claimed: {self.rows_claimed}")
        print(f"rows_skipped_already_claimed: {self.rows_skipped_already_claimed}")
        # Branch observability counters
        print(f"branch_steps_seen: {self.branch_steps_seen}")
        print(f"branch_steps_matched: {self.branch_steps_matched}")
        print(f"branch_steps_else_taken: {self.branch_steps_else_taken}")
        print(f"branch_steps_no_match_skipped: {self.branch_steps_no_match_skipped}")
        print(f"branch_parse_errors: {self.branch_parse_errors}")
        print(f"branch_dependency_skips: {self.branch_dependency_skips}")
        print("=" * 60)


async def run(project_id: str, run_id: str) -> None:
    """
    Run the enrichment workflow for a given project_id.
    
    Args:
        project_id: Project ID to scope all operations
        run_id: Run ID from the runs table (for checking if run is still active)
    """
    try:
        # Initialize workflow (creates global clients)
        workflow = EnrichmentWorkflow(project_id)
        # Run batch processing
        await workflow.run(run_id)
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        raise
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user. Progress has been saved.")
        raise
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise


async def main():
    """Entry point for the script."""
    # Get project_id from command line or environment variable
    import sys
    project_id = None
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
    else:
        project_id = os.getenv("PROJECT_ID")
    
    if not project_id:
        print("ERROR: project_id is required.")
        print("Usage: python enrich_workflow.py <project_id>")
        print("   OR: Set PROJECT_ID environment variable")
        sys.exit(1)
    
    try:
        # Generate a temporary run_id for direct script execution
        # (when not called from worker, there's no runs table entry)
        temp_run_id = str(uuid.uuid4())
        await run(project_id, temp_run_id)
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
