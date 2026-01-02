"""
Automated cold email workflow - Website enrichment script.

This script processes websites from Supabase, enriches them with Exa.AI summaries,
and categorizes them using OpenRouter LLM models.

Production-ready version with batch processing for large datasets.
"""

import asyncio
import os
import time
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

# LLM Configuration
LLM_MODEL_PRIMARY = "google/gemini-2.5-flash-lite-preview-09-2025"
LLM_MODEL_FALLBACK = "openai/gpt-5-mini"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 200
LLM_PROMPT_TEMPLATE = "Given a company description, I need you to confirm whether or not this company is primarily B2B or B2C. B2B is a company that provides software, services, or products to companies. A B2C company is a company that provides software, services, or products to consumers and not companies. If there is not enough information to generate an accurate answer, you must return an empty string with no words, otherwise you must only return a single sentence explaining your reasoning followed by either  - B2B or  - B2C. Please take your time to generate an accurate answer. Here is the company information: {exa_summary}"


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
    
    async def get_company_category(self, exa_summary: str) -> Optional[str]:
        """
        Get 2-word company category from OpenRouter LLM based on Exa summary.
        Uses primary model with automatic fallback to secondary model on failure.
        
        Args:
            exa_summary: The company summary from Exa.AI
            
        Returns:
            2-word category string or None if error occurs
        """
        # Format prompt using template
        prompt = LLM_PROMPT_TEMPLATE.format(exa_summary=exa_summary)
        
        # Try primary model first
        try:
            response = await self.openrouter_client.chat.completions.create(
                model=LLM_MODEL_PRIMARY,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE
            )
            
            if response and response.choices and len(response.choices) > 0:
                category = response.choices[0].message.content.strip()
                # Remove quotes if present
                category = category.strip('"\'')
                # Take first 2 words
                words = category.split()[:2]
                return ' '.join(words)
            
            return None
        except Exception as e:
            # Primary model failed, try fallback
            print(f"Primary model failed; retrying with fallback")
            try:
                response = await self.openrouter_client.chat.completions.create(
                    model=LLM_MODEL_FALLBACK,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE
                )
                
                if response and response.choices and len(response.choices) > 0:
                    category = response.choices[0].message.content.strip()
                    # Remove quotes if present
                    category = category.strip('"\'')
                    # Take first 2 words
                    words = category.split()[:2]
                    return ' '.join(words)
                
                return None
            except Exception as e2:
                # Fallback also failed
                print(f"Error getting company category from OpenRouter: {str(e2)}")
                return None
    
    async def enrich_prospect(self, prospect: Dict[str, Any], semaphore: asyncio.Semaphore) -> None:
        """
        Enrich a single prospect with Exa summary and OpenRouter category.
        
        Args:
            prospect: Dictionary containing prospect data (must have 'id' and 'website' fields)
            semaphore: Semaphore to limit concurrent requests
        """
        async with semaphore:
            prospect_id = prospect.get('id')
            website = prospect.get('website')
            
            if not website:
                print(f"  [ERROR] Prospect {prospect_id} has no website, marking as error")
                await self._update_prospect_status(prospect_id, 'error', None, None)
                return
            
            try:
                # Get Exa summary
                exa_summary = await self.get_exa_summary(website)
                
                if not exa_summary:
                    print(f"  [ERROR] No summary found for {website} (ID: {prospect_id}), marking as error")
                    await self._update_prospect_status(prospect_id, 'error', None, None)
                    return
                
                # Get company category from OpenRouter
                company_category = await self.get_company_category(exa_summary)
                
                if not company_category:
                    print(f"  [ERROR] No category found for {website} (ID: {prospect_id}), marking as error")
                    await self._update_prospect_status(prospect_id, 'error', exa_summary, None)
                    return
                
                # Update prospect with enriched data
                await self._update_prospect_status(
                    prospect_id, 
                    'enriched', 
                    exa_summary, 
                    company_category
                )
                
            except Exception as e:
                # Catch any unexpected errors and mark as error
                print(f"  [ERROR] Unexpected error processing prospect {prospect_id} ({website}): {str(e)}")
                await self._update_prospect_status(prospect_id, 'error', None, None)
    
    async def _update_prospect_status(
        self, 
        prospect_id: Any, 
        status: str, 
        exa_summary: Optional[str], 
        company_category: Optional[str]
    ) -> None:
        """
        Update prospect status and enrichment data in Supabase.
        
        Args:
            prospect_id: The ID of the prospect to update
            status: New status ('enriched' or 'error')
            exa_summary: The summary from Exa.AI (can be None)
            company_category: The category from OpenRouter (can be None)
        """
        try:
            update_data = {'status': status}
            
            if exa_summary is not None:
                update_data['exa_summary'] = exa_summary
            
            if company_category is not None:
                update_data['company_category'] = company_category
            
            # Use asyncio.to_thread for the synchronous Supabase call (using global client)
            def update_prospect():
                return (
                    self.supabase.table('prospects')
                    .update(update_data)
                    .eq('id', prospect_id)
                    .execute()
                )
            
            await asyncio.to_thread(update_prospect)
            
        except Exception as e:
            print(f"  [ERROR] Failed to update prospect {prospect_id} in database: {str(e)}")
    
    def fetch_batch(self, limit: int = BATCH_SIZE) -> list:
        """
        Fetch a batch of prospects with status 'new' from Supabase.
        
        Args:
            limit: Number of rows to fetch
            
        Returns:
            List of prospect dictionaries
        """
        try:
            response = (
                self.supabase.table('prospects')
                .select('*')
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
        # Capture start time at the very beginning
        start_time = time.time()
        
        print("=" * 60)
        print("Starting enrichment workflow (Production Mode)")
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
        print(f"  ✓ Semaphore limit: {SEMAPHORE_LIMIT} concurrent requests")
        print(f"  ✓ Batch size: {BATCH_SIZE} rows per batch")
        
        # Data cleaning phase
        print("\n[3/4] Running data cleaning phase...")
        await self.clean_data()
        
        print("\n[4/4] Starting batch processing...")
        print("-" * 60)
        
        # Create semaphore for rate limiting (shared across all batches)
        semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
        
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
            
            print(f"[Batch {batch_number}] Fetched {len(batch)} prospects. Processing...")
            
            # Process this batch concurrently with semaphore
            tasks = [
                self.enrich_prospect(prospect, semaphore)
                for prospect in batch
            ]
            
            # Wait for all tasks in this batch to complete
            await asyncio.gather(*tasks)
            
            total_processed += len(batch)
            print(f"[Batch {batch_number}] Batch Complete: {len(batch)} rows processed (Total: {total_processed})")
        
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
        print(f"✓ Total prospects processed: {total_processed}")
        print(f"✓ Total time elapsed: {duration_minutes} min {duration_secs} sec")
        print(f"✓ Average speed: {avg_seconds_per_prospect:.2f} seconds per prospect")
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
