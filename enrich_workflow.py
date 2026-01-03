"""
Automated cold email workflow - Website enrichment script.

This script processes websites from Supabase, enriches them with Exa.AI summaries,
and categorizes them using OpenRouter LLM models.

Production-ready version with batch processing for large datasets.
"""

import asyncio
import os
import time
import uuid
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from exa_py import Exa
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SEMAPHORE_LIMIT = 5  # Max concurrent requests (respects Exa's 10 QPS limit)
BATCH_SIZE = 50  # Number of rows to fetch and process per batch

# Multi-step AI Enrichment Configuration
NUM_STEPS_TO_RUN = 2  # Number of enrichment steps to execute (expandable to 4)

# Step 1 Configuration
STEP1_MODEL_PRIMARY = "google/gemini-2.5-flash"
STEP1_MODEL_FALLBACK = "openai/gpt-5-mini"
STEP1_TEMPERATURE = 0.3
STEP1_MAX_TOKENS = 250
STEP1_PROMPT_TEMPLATE = "Give me a summary of this company: {exa_summary}"

# Step 2 Configuration
STEP2_MODEL_PRIMARY = "google/gemini-2.5-flash"
STEP2_MODEL_FALLBACK = "openai/gpt-5-mini"
STEP2_TEMPERATURE = 0.3
STEP2_MAX_TOKENS = 250
STEP2_PROMPT_TEMPLATE = "Given this company description: {exa_summary}. And this prior result: {step1_output}. Provide one sentence expanding on the category."

# Step 3 Configuration (for future expansion)
STEP3_MODEL_PRIMARY = "google/gemini-2.5-flash"
STEP3_MODEL_FALLBACK = "openai/gpt-5-mini"
STEP3_TEMPERATURE = 0.3
STEP3_MAX_TOKENS = 250
STEP3_PROMPT_TEMPLATE = "Is this company B2B or B2C?: {exa_summary}"

# Step 4 Configuration (for future expansion)
STEP4_MODEL_PRIMARY = "google/gemini-2.5-flash"
STEP4_MODEL_FALLBACK = "openai/gpt-5-mini"
STEP4_TEMPERATURE = 0.3
STEP4_MAX_TOKENS = 250
STEP4_PROMPT_TEMPLATE = "Generate a list of this companys services: {exa_summary}"


class EnrichmentWorkflow:
    """Handles the enrichment workflow for prospect websites."""
    
    def __init__(self):
        """Initialize global clients for Supabase, Exa.AI, and OpenRouter (reused across all batches)."""
        if not all([SUPABASE_URL, SUPABASE_KEY, EXA_API_KEY, OPENROUTER_API_KEY]):
            raise ValueError(
                "Missing required environment variables. "
                "Please check your .env file for: "
                "SUPABASE_URL, SUPABASE_KEY, EXA_API_KEY, OPENROUTER_API_KEY"
            )
        
        # Initialize global Supabase client (reused across all batches)
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Initialize global Exa.AI client (reused across all batches)
        self.exa = Exa(api_key=EXA_API_KEY)
        
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
        
        # Run ID for idempotency within this run
        self.run_id = None
    
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
        exa_summary: str,
        step_outputs: Dict[int, str]
    ) -> Optional[str]:
        """
        Run a single enrichment step.
        
        Args:
            step_num: Step number (1-4)
            exa_summary: The Exa summary
            step_outputs: Dictionary of previous step outputs (key: step_num, value: output)
            
        Returns:
            Step output or None if failed
        """
        # Get step configuration
        config_map = {
            1: {
                'model_primary': STEP1_MODEL_PRIMARY,
                'model_fallback': STEP1_MODEL_FALLBACK,
                'temperature': STEP1_TEMPERATURE,
                'max_tokens': STEP1_MAX_TOKENS,
                'prompt_template': STEP1_PROMPT_TEMPLATE
            },
            2: {
                'model_primary': STEP2_MODEL_PRIMARY,
                'model_fallback': STEP2_MODEL_FALLBACK,
                'temperature': STEP2_TEMPERATURE,
                'max_tokens': STEP2_MAX_TOKENS,
                'prompt_template': STEP2_PROMPT_TEMPLATE
            },
            3: {
                'model_primary': STEP3_MODEL_PRIMARY,
                'model_fallback': STEP3_MODEL_FALLBACK,
                'temperature': STEP3_TEMPERATURE,
                'max_tokens': STEP3_MAX_TOKENS,
                'prompt_template': STEP3_PROMPT_TEMPLATE
            },
            4: {
                'model_primary': STEP4_MODEL_PRIMARY,
                'model_fallback': STEP4_MODEL_FALLBACK,
                'temperature': STEP4_TEMPERATURE,
                'max_tokens': STEP4_MAX_TOKENS,
                'prompt_template': STEP4_PROMPT_TEMPLATE
            }
        }
        
        config = config_map[step_num]
        
        # Format prompt with placeholders
        prompt = config['prompt_template'].format(
            exa_summary=exa_summary,
            step1_output=step_outputs.get(1, ''),
            step2_output=step_outputs.get(2, ''),
            step3_output=step_outputs.get(3, ''),
            step4_output=step_outputs.get(4, '')
        )
        
        # Call LLM with retry
        return await self._call_llm_with_retry(
            step_num=step_num,
            prompt=prompt,
            model_primary=config['model_primary'],
            model_fallback=config['model_fallback'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
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
                    .select('id, status, exa_summary, step1_output, step2_output, step3_output, step4_output')
                    .eq('id', prospect_id)
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
            
            # Verify step outputs
            for step_num in range(1, NUM_STEPS_TO_RUN + 1):
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
        step_outputs: Dict[int, str],
        db_write_semaphore: asyncio.Semaphore
    ) -> bool:
        """
        Write prospect data to Supabase with retry logic and verification.
        
        Args:
            prospect_id: The ID of the prospect to update
            exa_summary: The summary from Exa.AI (can be None)
            step_outputs: Dictionary of step outputs (key: step_num, value: output)
            db_write_semaphore: Semaphore to serialize DB writes (concurrency=1)
            
        Returns:
            True if write succeeded and was verified, False otherwise
        """
        async with db_write_semaphore:
            self.db_writes_attempted += 1
            
            # Build update data
            update_data = {'status': 'enriched'}
            
            if exa_summary is not None:
                update_data['exa_summary'] = exa_summary
            
            # Only include step outputs for required steps (1..NUM_STEPS_TO_RUN)
            for step_num in range(1, NUM_STEPS_TO_RUN + 1):
                step_key = f'step{step_num}_output'
                if step_num in step_outputs and step_outputs[step_num]:
                    update_data[step_key] = step_outputs[step_num]
            
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
        
        # Check if prospect is eligible (has at least one blank required step)
        has_blank_step = False
        for step_num in range(1, NUM_STEPS_TO_RUN + 1):
            step_key = f'step{step_num}_output'
            if step_key not in prospect or not prospect[step_key]:
                has_blank_step = True
                break
        
        if not has_blank_step:
            # All required steps already have output, skip this prospect
            return
        
        async with ai_semaphore:
            # In-memory cache for AI results
            cached_exa_summary: Optional[str] = None
            cached_step_outputs: Dict[int, str] = {}
            
            try:
                # Get existing step outputs from prospect data
                for step in range(1, 5):
                    step_key = f'step{step}_output'
                    if step_key in prospect and prospect[step_key]:
                        cached_step_outputs[step] = prospect[step_key]
                
                # REQUIRE EXA SUMMARY BEFORE STEP 1
                # Get Exa summary (use existing if available and non-empty, otherwise fetch)
                existing_exa_summary = prospect.get('exa_summary')
                if existing_exa_summary and existing_exa_summary.strip():
                    cached_exa_summary = existing_exa_summary
                else:
                    # Fetch Exa summary
                    cached_exa_summary = await self.get_exa_summary(website)
                
                # If Exa summary is still empty/None after fetch, this is a final failure
                if not cached_exa_summary or not cached_exa_summary.strip():
                    print(f"  [EXA FAILURE] No summary found for {website} (ID: {prospect_id}, normalized: {normalized_website}, run_id: {self.run_id})")
                    self.exa_failures += 1
                    failed_this_run.add(normalized_website)
                    return
                
                # Run all required enrichment steps sequentially, caching outputs in memory
                for step_num in range(1, NUM_STEPS_TO_RUN + 1):
                    # Skip if this step already has output
                    if step_num in cached_step_outputs and cached_step_outputs[step_num]:
                        continue
                    
                    # Ensure all previous required steps have outputs
                    all_previous_complete = True
                    for prev_step in range(1, step_num):
                        if prev_step not in cached_step_outputs or not cached_step_outputs[prev_step]:
                            all_previous_complete = False
                            break
                    
                    if not all_previous_complete:
                        print(f"  [ERROR] Prospect {prospect_id} (run_id: {self.run_id}): Step {step_num} cannot run - previous steps incomplete")
                        failed_this_run.add(normalized_website)
                        return
                    
                    # Run the step
                    step_result = await self._run_enrichment_step(step_num, cached_exa_summary, cached_step_outputs)
                    
                    if not step_result:
                        # Step failed after all retries - this is a final failure
                        print(f"  [ERROR] Prospect {prospect_id} (normalized: {normalized_website}, run_id: {self.run_id}): Step {step_num} failed after all retries")
                        failed_this_run.add(normalized_website)
                        return
                    
                    # Store the result in memory cache
                    cached_step_outputs[step_num] = step_result
                
                # Verify all required steps are complete
                all_steps_complete = True
                for step_num in range(1, NUM_STEPS_TO_RUN + 1):
                    if step_num not in cached_step_outputs or not cached_step_outputs[step_num]:
                        all_steps_complete = False
                        break
                
                if not all_steps_complete:
                    print(f"  [ERROR] Prospect {prospect_id} (run_id: {self.run_id}): Not all required steps completed")
                    failed_this_run.add(normalized_website)
                    return
                
                # All steps succeeded - perform single final write to Supabase
                write_success = await self._write_prospect_with_retry(
                    prospect_id,
                    cached_exa_summary,
                    cached_step_outputs,
                    db_write_semaphore
                )
                
                if write_success:
                    self.prospects_enriched += 1
                    # Immediately discard cached data (memory safety)
                    cached_exa_summary = None
                    cached_step_outputs = {}
                else:
                    # DB write failed after all retries - this is a final failure
                    self.prospects_deferred_db_write += 1
                    print(f"  [DEFERRED] Prospect {prospect_id} (normalized: {normalized_website}, run_id: {self.run_id}): All retries failed. Row remains status='new' for next run.")
                    failed_this_run.add(normalized_website)
                    # Discard cached data even on failure (memory safety)
                    cached_exa_summary = None
                    cached_step_outputs = {}
                
            except Exception as e:
                # Catch any unexpected errors - treat as final failure
                print(f"  [ERROR] Unexpected error processing prospect {prospect_id} ({website}, normalized: {normalized_website}, run_id: {self.run_id}): {str(e)}")
                failed_this_run.add(normalized_website)
                # Discard cached data on error (memory safety)
                cached_exa_summary = None
                cached_step_outputs = {}
    
    def fetch_batch(self, limit: int = BATCH_SIZE) -> list:
        """
        Fetch a batch of prospects with status 'new' from Supabase.
        Also fetch existing step outputs to support "only fill blanks" logic.
        
        Args:
            limit: Number of rows to fetch
            
        Returns:
            List of prospect dictionaries
        """
        try:
            response = (
                self.supabase.table('prospects')
                .select('id, website, exa_summary, step1_output, step2_output, step3_output, step4_output')
                .eq('status', 'new')
                .limit(limit)
                .execute()
            )
            
            return response.data if response.data else []
        except Exception as e:
            print(f"Error fetching prospects batch: {str(e)}")
            return []
    
    async def run(self) -> None:
        """
        Main workflow execution with batch processing.
        
        Fetches 50 rows at a time, processes them, then fetches the next batch.
        Continues until no more rows with status='new' are found.
        """
        # Generate run_id for idempotency within this run
        self.run_id = str(uuid.uuid4())[:8]
        
        # Capture start time at the very beginning
        start_time = time.time()
        
        print("=" * 60)
        print("Starting enrichment workflow (Production Mode)")
        print(f"Run ID: {self.run_id}")
        print("=" * 60)
        
        # Test Supabase connection first
        print("\n[1/4] Testing Supabase connection...")
        connection_ok = await self.test_supabase_connection()
        if not connection_ok:
            print("Cannot proceed without a valid Supabase connection.")
            return
        
        print("\n[2/4] Initializing clients...")
        print(f"  ✓ Supabase client initialized")
        print(f"  ✓ Exa.AI client initialized")
        print(f"  ✓ OpenRouter client initialized")
        print(f"  ✓ AI Semaphore limit: {SEMAPHORE_LIMIT} concurrent requests")
        print(f"  ✓ DB Write Semaphore: 1 (serialized writes)")
        print(f"  ✓ Batch size: {BATCH_SIZE} rows per batch")
        print(f"  ✓ Number of enrichment steps: {NUM_STEPS_TO_RUN}")
        
        # Data cleaning phase
        print("\n[3/4] Running data cleaning phase...")
        await self.clean_data()
        
        print("\n[4/4] Starting batch processing...")
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
            
            # Fetch next batch of 50 rows
            print(f"\n[Batch {batch_number}] Fetching {BATCH_SIZE} prospects with status='new'...")
            batch = self.fetch_batch(BATCH_SIZE)
            
            # If no rows returned, we're done
            if not batch:
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
        print(f"exa_failures: {self.exa_failures}")
        print(f"ai_calls_attempted: {self.ai_calls_attempted}")
        print(f"ai_calls_retried: {self.ai_calls_retried}")
        print(f"db_writes_attempted: {self.db_writes_attempted}")
        print(f"db_write_retries: {self.db_write_retries}")
        print(f"db_write_failures: {self.db_write_failures}")
        print("=" * 60)


async def main():
    """Entry point for the script."""
    try:
        # Initialize workflow (creates global clients)
        workflow = EnrichmentWorkflow()
        # Run batch processing
        await workflow.run()
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
