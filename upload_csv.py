"""
Bulk upload script for prospect websites.

This script reads websites from input.csv and uploads them to the Supabase prospects table,
skipping duplicates. It also uploads prompts from the Prompts column to the public.prompts table.
"""

import csv
import os
import sys
from typing import Set, List, Dict, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
INPUT_CSV = "input.csv"


class CSVUploader:
    """Handles CSV upload to Supabase prospects table and prompts table."""
    
    def __init__(self):
        """Initialize Supabase client."""
        if not all([SUPABASE_URL, SUPABASE_KEY]):
            raise ValueError(
                "Missing required environment variables. "
                "Please check your .env file for: "
                "SUPABASE_URL, SUPABASE_KEY"
            )
        
        # Initialize Supabase client
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.has_prompts_column: Optional[bool] = None
    
    def test_connection(self) -> bool:
        """
        Test the Supabase connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = (
                self.supabase.table('prospects')
                .select('id')
                .limit(1)
                .execute()
            )
            print("✓ Successfully connected to Supabase!")
            return True
        except Exception as e:
            print(f"✗ Error connecting to Supabase: {str(e)}")
            return False
    
    def check_prompts_column(self) -> bool:
        """
        Check if the prospects table has a 'prompts' column.
        
        Returns:
            True if column exists, False otherwise
        """
        if self.has_prompts_column is not None:
            return self.has_prompts_column
        
        try:
            # Try to select prompts column - if it doesn't exist, this will fail
            response = (
                self.supabase.table('prospects')
                .select('prompts')
                .limit(1)
                .execute()
            )
            self.has_prompts_column = True
            return True
        except Exception as e:
            # If error mentions column doesn't exist, mark as False
            error_str = str(e).lower()
            if 'column' in error_str and ('does not exist' in error_str or 'not found' in error_str):
                self.has_prompts_column = False
                return False
            # For other errors, assume column exists (safer to try)
            self.has_prompts_column = True
            return True
    
    def extract_prompts_from_csv(self, csv_path: str) -> tuple[List[Dict[str, str]], int]:
        """
        Extract and normalize prompts and run_if from the Prompts and Run If columns in the CSV.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (deduplicated prompts list with run_if, raw count before deduplication)
            Each item in the list is a dict with 'prompt_text' and 'run_if' keys
        """
        prompts_raw = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check if Prompts column exists
                if reader.fieldnames is None:
                    raise ValueError("CSV file appears to be empty or invalid.")
                
                if 'Prompts' not in reader.fieldnames:
                    raise ValueError(
                        f"CSV file is missing required header: 'Prompts'. "
                        f"Found columns: {', '.join(reader.fieldnames)}"
                    )
                
                # Collect all non-empty prompts with their run_if values
                for row in reader:
                    prompt = row.get('Prompts', '').strip()
                    if prompt:  # Only add non-empty prompts after stripping
                        run_if = row.get('Run If', '').strip()
                        prompts_raw.append({
                            'prompt_text': prompt,
                            'run_if': run_if if run_if else None
                        })
            
            prompts_found_raw = len(prompts_raw)
            
            # Deduplicate while preserving order (keep first occurrence's run_if)
            prompts_seen = set()
            prompts_deduplicated = []
            for prompt_data in prompts_raw:
                prompt_text = prompt_data['prompt_text']
                if prompt_text not in prompts_seen:
                    prompts_seen.add(prompt_text)
                    prompts_deduplicated.append(prompt_data)
            
            return prompts_deduplicated, prompts_found_raw
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise Exception(f"Error reading prompts from CSV file: {str(e)}")
    
    def upload_prompts(self, prompts: List[Dict[str, str]]) -> None:
        """
        Replace all prompts in public.prompts table with the new prompt list.
        
        Args:
            prompts: List of prompt dictionaries with 'prompt_text' and 'run_if' keys
            
        Raises:
            Exception: If upload fails
        """
        try:
            # Delete all existing prompts
            # Use a condition that matches all rows (step_order >= 0 should match all valid rows)
            delete_response = (
                self.supabase.table('prompts')
                .delete()
                .gte('step_order', 0)
                .execute()
            )
            
            # If no prompts to insert, we're done
            if not prompts:
                return
            
            # Prepare insert data
            insert_data = []
            for idx, prompt_data in enumerate(prompts, start=1):
                insert_data.append({
                    'step_order': idx,
                    'prompt_text': prompt_data['prompt_text'],
                    'run_if': prompt_data.get('run_if'),
                    'is_active': True,
                    'step_name': f'step{idx}'
                })
            
            # Insert new prompts
            insert_response = (
                self.supabase.table('prompts')
                .insert(insert_data)
                .execute()
            )
            
        except Exception as e:
            raise Exception(f"Error uploading prompts to database: {str(e)}")
    
    def get_existing_websites(self) -> Set[str]:
        """
        Fetch all existing websites from the prospects table.
        
        Returns:
            Set of existing website strings (normalized)
        """
        try:
            existing_websites = set()
            offset = 0
            batch_size = 1000
            
            while True:
                response = (
                    self.supabase.table('prospects')
                    .select('website')
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )
                
                if not response.data:
                    break
                
                # Normalize websites
                for row in response.data:
                    website = row.get('website')
                    if website:
                        normalized = self._normalize_website(website)
                        existing_websites.add(normalized)
                
                if len(response.data) < batch_size:
                    break
                
                offset += batch_size
            
            return existing_websites
        except Exception as e:
            print(f"Error fetching existing websites: {str(e)}")
            return set()
    
    def _normalize_website(self, website: str) -> str:
        """
        Normalize website URL: remove protocol, www, trailing slashes, convert to lowercase.
        Matches normalization logic from enrich_workflow.py.
        
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
    
    def read_csv_rows(self, csv_path: str) -> List[Dict[str, str]]:
        """
        Read rows from CSV file with new format.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of dictionaries with 'website', 'short_description' keys
        """
        rows = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check required headers
                if reader.fieldnames is None:
                    raise ValueError("CSV file appears to be empty or invalid.")
                
                required_headers = ['Website']
                missing_headers = [h for h in required_headers if h not in reader.fieldnames]
                
                if missing_headers:
                    raise ValueError(
                        f"CSV file is missing required headers: {', '.join(missing_headers)}. "
                        f"Found columns: {', '.join(reader.fieldnames)}"
                    )
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                    website = row.get('Website', '').strip()
                    short_description = row.get('Short Description', '').strip()
                    
                    rows.append({
                        'website': website,
                        'short_description': short_description if short_description else None
                    })
            
            return rows
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
    
    def upload_rows(self, rows: List[Dict[str, str]], existing_websites: Set[str]) -> tuple[int, int, int, int]:
        """
        Upload new rows to Supabase, skipping duplicates.
        
        Args:
            rows: List of row dictionaries with website, short_description
            existing_websites: Set of normalized existing websites
            
        Returns:
            Tuple of (inserted_count, skipped_blank_count, skipped_duplicate_count, failed_count)
        """
        inserted_count = 0
        skipped_blank_count = 0
        skipped_duplicate_count = 0
        failed_count = 0
        new_rows = []
        
        # Filter out duplicates and blank websites
        for row in rows:
            website = row['website']
            
            # Skip blank websites
            if not website:
                skipped_blank_count += 1
                continue
            
            normalized = self._normalize_website(website)
            
            # Skip duplicates
            if normalized in existing_websites:
                skipped_duplicate_count += 1
            else:
                # Prepare insert data
                insert_data = {
                    'website': normalized,
                    'short_description': row['short_description'],
                    'status': 'new'
                }
                
                new_rows.append(insert_data)
                # Add to existing set to avoid duplicates within the same batch
                existing_websites.add(normalized)
        
        # Batch insert new rows
        if new_rows:
            try:
                # Insert in batches to avoid payload size limits
                batch_size = 100
                for i in range(0, len(new_rows), batch_size):
                    batch = new_rows[i:i + batch_size]
                    try:
                        response = (
                            self.supabase.table('prospects')
                            .insert(batch)
                            .execute()
                        )
                        inserted_count += len(batch)
                        print(f"Uploaded batch: {len(batch)} rows (total: {inserted_count})")
                    except Exception as e:
                        print(f"Error uploading batch: {str(e)}")
                        failed_count += len(batch)
            except Exception as e:
                print(f"Error during batch upload: {str(e)}")
                failed_count += len(new_rows) - inserted_count
        
        return inserted_count, skipped_blank_count, skipped_duplicate_count, failed_count
    
    def run(self) -> None:
        """Main upload workflow execution."""
        print("Starting CSV upload workflow...")
        print("-" * 50)
        
        # Test connection
        print("Testing Supabase connection...")
        if not self.test_connection():
            print("Cannot proceed without a valid Supabase connection.")
            sys.exit(1)
        
        print("-" * 50)
        
        # Check if CSV file exists
        if not os.path.exists(INPUT_CSV):
            print(f"Error: {INPUT_CSV} not found in the current directory.")
            sys.exit(1)
        
        # Extract and upload prompts FIRST (before prospect upload)
        print("Extracting prompts from CSV...")
        try:
            prompts_deduplicated, prompts_found_raw = self.extract_prompts_from_csv(INPUT_CSV)
            prompts_inserted = len(prompts_deduplicated)
            
            print(f"prompts_found_raw: {prompts_found_raw}")
            print(f"prompts_inserted: {prompts_inserted}")
            
            # If zero prompts found, log error and exit
            if prompts_inserted == 0:
                print("ERROR: Zero prompts found in CSV. Aborting to prevent accidental deletion of prompts table.")
                sys.exit(1)
            
            # Upload prompts
            print("Uploading prompts to public.prompts...")
            self.upload_prompts(prompts_deduplicated)
            print(f"Deleted existing prompts and inserted {prompts_inserted} new prompts (with run_if).")
            
        except Exception as e:
            print(f"ERROR: Failed to upload prompts: {str(e)}")
            sys.exit(1)
        
        print("-" * 50)
        
        # Read rows from CSV for prospects
        print(f"Reading rows from {INPUT_CSV}...")
        try:
            rows = self.read_csv_rows(INPUT_CSV)
            rows_read = len(rows)
            print(f"Found {rows_read} rows in CSV")
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            return
        
        if not rows:
            print("No rows found in CSV file.")
            return
        
        print("-" * 50)
        
        # Fetch existing websites
        print("Checking for existing websites in database...")
        existing_websites = self.get_existing_websites()
        print(f"Found {len(existing_websites)} existing websites in database")
        
        print("-" * 50)
        
        # Upload new rows
        print("Uploading new rows...")
        inserted_count, skipped_blank_count, skipped_duplicate_count, failed_count = self.upload_rows(rows, existing_websites)
        
        print("-" * 50)
        print("Upload Summary:")
        print(f"  Rows read: {rows_read}")
        print(f"  Rows skipped (blank website): {skipped_blank_count}")
        print(f"  Rows skipped (duplicates): {skipped_duplicate_count}")
        print(f"  Rows inserted: {inserted_count}")
        print(f"  Rows failed: {failed_count}")
        print("Upload workflow completed!")


def main():
    """Entry point for the script."""
    try:
        uploader = CSVUploader()
        uploader.run()
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
