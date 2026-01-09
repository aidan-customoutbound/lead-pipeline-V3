"""
Bulk upload script for prospect websites.

This script reads websites from input.csv and uploads them to the Supabase prospects table,
skipping duplicates. It also uploads prompts from the Prompts column to the public.prompts table.
"""

import csv
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Set
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
    
    def __init__(self, project_id: str):
        """
        Initialize Supabase client.
        
        Args:
            project_id: Project ID to scope all operations (required)
        """
        if not project_id:
            raise ValueError("project_id is required and cannot be empty")
        
        if not all([SUPABASE_URL, SUPABASE_KEY]):
            raise ValueError(
                "Missing required environment variables. "
                "Please check your .env file for: "
                "SUPABASE_URL, SUPABASE_KEY"
            )
        
        # Initialize Supabase client
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.has_prompts_column: Optional[bool] = None
        self.project_id = project_id
    
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
    
    def extract_prompts_from_rows(self, csv_rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], int]:
        """
        Extract and normalize prompts, run_if, and branch from the Prompts, Run If, and Branch columns in CSV rows.
        
        Supports "Branch-only" steps: a row is considered a valid prompt step if it has:
        - non-blank Prompts OR
        - non-blank Branch
        
        Args:
            csv_rows: List of CSV row dictionaries (from DictReader)
            
        Returns:
            Tuple of (deduplicated prompts list with run_if and branch, raw count before deduplication)
            Each item in the list is a dict with 'prompt_text', 'run_if', and 'branch' keys
        """
        prompts_raw = []
        BRANCH_PLACEHOLDER = "BRANCH_CONTROLLED"
        
        if not csv_rows:
            raise ValueError("CSV rows list is empty or invalid.")
        
        # Get fieldnames from first row keys
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        
        # Check if Prompts column exists (required for backward compatibility check)
        # But we'll also accept Branch-only rows
        has_prompts_column = 'Prompts' in fieldnames
        has_branch_column = 'Branch' in fieldnames
        
        if not has_prompts_column and not has_branch_column:
            raise ValueError(
                f"CSV rows must have at least one of: 'Prompts' or 'Branch' columns. "
                f"Found columns: {', '.join(fieldnames)}"
            )
        
        # Collect all valid prompt steps (Prompts OR Branch non-empty)
        for row in csv_rows:
            prompt = row.get('Prompts', '').strip() if has_prompts_column else ''
            branch = row.get('Branch', '').strip() if has_branch_column else ''
            
            # Row is valid if Prompts OR Branch is non-empty
            if prompt or branch:
                run_if = row.get('Run If', '').strip() if 'Run If' in fieldnames else ''
                
                # If Prompts is blank but Branch is present, use placeholder
                if not prompt and branch:
                    prompt_text = BRANCH_PLACEHOLDER
                else:
                    prompt_text = prompt
                
                prompts_raw.append({
                    'prompt_text': prompt_text,
                    'run_if': run_if if run_if else None,
                    'branch': branch if branch else None
                })
        
        prompts_found_raw = len(prompts_raw)
        
        # Deduplicate while preserving order (dedupe on tuple (prompt_text, run_if, branch))
        prompts_seen = set()
        prompts_deduplicated = []
        for prompt_data in prompts_raw:
            # Create dedupe key from tuple (prompt_text, run_if, branch)
            dedupe_key = (
                prompt_data['prompt_text'],
                prompt_data.get('run_if'),
                prompt_data.get('branch')
            )
            if dedupe_key not in prompts_seen:
                prompts_seen.add(dedupe_key)
                prompts_deduplicated.append(prompt_data)
        
        return prompts_deduplicated, prompts_found_raw
    
    def extract_prompts_from_csv(self, csv_path: str) -> tuple[List[Dict[str, str]], int]:
        """
        Extract and normalize prompts, run_if, and branch from the Prompts, Run If, and Branch columns in the CSV.
        
        Supports "Branch-only" steps: a row is considered a valid prompt step if it has:
        - non-blank Prompts OR
        - non-blank Branch
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (deduplicated prompts list with run_if and branch, raw count before deduplication)
            Each item in the list is a dict with 'prompt_text', 'run_if', and 'branch' keys
        """
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check if CSV is valid
                if reader.fieldnames is None:
                    raise ValueError("CSV file appears to be empty or invalid.")
                
                # Convert reader to list of dicts
                csv_rows = list(reader)
                
                # Use the row-based extraction method
                return self.extract_prompts_from_rows(csv_rows)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise Exception(f"Error reading prompts from CSV file: {str(e)}")
    
    def upload_prompts(self, prompts: List[Dict[str, str]]) -> None:
        """
        Replace all prompts in public.prompts table with the new prompt list.
        
        Args:
            prompts: List of prompt dictionaries with 'prompt_text', 'run_if', and 'branch' keys
            
        Raises:
            Exception: If upload fails
        """
        try:
            # Delete all existing prompts for this project_id
            delete_response = (
                self.supabase.table('prompts')
                .delete()
                .eq('project_id', self.project_id)
                .execute()
            )
            
            # If no prompts to insert, we're done
            if not prompts:
                return
            
            # Prepare insert data
            insert_data = []
            for idx, prompt_data in enumerate(prompts, start=1):
                insert_row = {
                    'step_order': idx,
                    'prompt_text': prompt_data['prompt_text'],
                    'run_if': prompt_data.get('run_if'),
                    'is_active': True,
                    'step_name': f'step{idx}',
                    'project_id': self.project_id
                }
                # Include branch if present
                if 'branch' in prompt_data and prompt_data.get('branch'):
                    insert_row['branch'] = prompt_data['branch']
                insert_data.append(insert_row)
            
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
        Fetch all existing websites from the prospects table for this project_id.
        
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
                    .eq('project_id', self.project_id)
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
    
    def _is_exa_blank(self, exa_value: str) -> bool:
        """
        Check if Exa value is blank.
        
        Blank means:
        1) empty string ""
        2) whitespace-only
        3) case-insensitive exact match of "NULL"
        4) case-insensitive exact match of "N/A"
        
        Args:
            exa_value: The Exa value to check
            
        Returns:
            True if blank, False otherwise
        """
        if not exa_value:
            return True
        
        # Strip whitespace
        stripped = exa_value.strip()
        
        # Check for empty or whitespace-only
        if not stripped:
            return True
        
        # Check for case-insensitive "NULL" or "N/A"
        stripped_lower = stripped.lower()
        if stripped_lower == "null" or stripped_lower == "n/a":
            return True
        
        return False
    
    def parse_csv_rows(self, csv_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Parse CSV rows into prospect format.
        
        Args:
            csv_rows: List of CSV row dictionaries (from DictReader)
            
        Returns:
            List of dictionaries with 'website', 'short_description', 'exa' keys
        """
        rows = []
        
        if not csv_rows:
            raise ValueError("CSV rows list is empty or invalid.")
        
        # Get fieldnames from first row keys
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        
        # Check required headers
        required_headers = ['Website']
        missing_headers = [h for h in required_headers if h not in fieldnames]
        
        if missing_headers:
            raise ValueError(
                f"CSV rows are missing required headers: {', '.join(missing_headers)}. "
                f"Found columns: {', '.join(fieldnames)}"
            )
        
        for row in csv_rows:
            website = row.get('Website', '').strip()
            short_description = row.get('Short Description', '').strip()
            exa = row.get('Exa', '').strip() if 'Exa' in row else ''
            
            rows.append({
                'website': website,
                'short_description': short_description if short_description else None,
                'exa': exa
            })
        
        return rows
    
    def read_csv_rows(self, csv_path: str) -> List[Dict[str, str]]:
        """
        Read rows from CSV file with new format.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of dictionaries with 'website', 'short_description', 'exa' keys
        """
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check required headers
                if reader.fieldnames is None:
                    raise ValueError("CSV file appears to be empty or invalid.")
                
                # Convert reader to list of dicts
                csv_rows = list(reader)
                
                # Use the row-based parsing method
                return self.parse_csv_rows(csv_rows)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
    
    def upload_rows(self, rows: List[Dict[str, str]], existing_websites: Set[str], run_token: str) -> tuple[int, int, int, int, int, int]:
        """
        Upload new rows to Supabase, skipping duplicates.
        Also updates existing rows with exa_summary if CSV Exa is non-blank.
        
        Args:
            rows: List of row dictionaries with website, short_description, exa
            existing_websites: Set of normalized existing websites (project-specific)
            run_token: Run token to attach to all inserted rows
            
        Returns:
            Tuple of (inserted_count, skipped_blank_count, skipped_duplicate_count, failed_count, exa_overwrites_applied, exa_values_used_for_new_rows)
        """
        inserted_count = 0
        skipped_blank_count = 0
        skipped_duplicate_count = 0
        failed_count = 0
        exa_overwrites_applied = 0
        exa_values_used_for_new_rows = 0
        new_rows = []
        updates_to_apply = []  # List of (normalized_website, exa_summary) tuples
        
        # Filter out duplicates and blank websites, collect Exa updates
        for row in rows:
            website = row['website']
            
            # Skip blank websites
            if not website:
                skipped_blank_count += 1
                continue
            
            normalized = self._normalize_website(website)
            exa_value = row.get('exa', '')
            
            # Check if Exa is non-blank
            exa_is_blank = self._is_exa_blank(exa_value)
            exa_summary = None
            if not exa_is_blank:
                exa_summary = exa_value.strip()
            
            # Check if this is a duplicate
            if normalized in existing_websites:
                skipped_duplicate_count += 1
                # If Exa is non-blank, schedule an update for this existing row
                if not exa_is_blank:
                    updates_to_apply.append((normalized, exa_summary))
            else:
                # Prepare insert data
                insert_data = {
                    'website': normalized,
                    'short_description': row['short_description'],
                    'status': 'new',
                    'project_id': self.project_id,
                    'run_token': run_token
                }
                
                # Include exa_summary if Exa is non-blank
                if not exa_is_blank:
                    insert_data['exa_summary'] = exa_summary
                    exa_values_used_for_new_rows += 1
                
                new_rows.append(insert_data)
                # Add to existing set to avoid duplicates within the same batch
                existing_websites.add(normalized)
        
        # Apply updates to existing rows with non-blank Exa
        if updates_to_apply:
            print(f"Applying {len(updates_to_apply)} Exa overwrites to existing rows...")
            for normalized_website, exa_summary in updates_to_apply:
                try:
                    # Update the existing row by normalized website and project_id
                    response = (
                        self.supabase.table('prospects')
                        .update({'exa_summary': exa_summary})
                        .eq('website', normalized_website)
                        .eq('project_id', self.project_id)
                        .execute()
                    )
                    exa_overwrites_applied += 1
                except Exception as e:
                    print(f"Error updating exa_summary for {normalized_website}: {str(e)}")
        
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
        
        return inserted_count, skipped_blank_count, skipped_duplicate_count, failed_count, exa_overwrites_applied, exa_values_used_for_new_rows
    
    def run(self) -> None:
        """Main upload workflow execution."""
        # Generate run token at the start
        current_run_token = str(uuid.uuid4())
        
        print("Starting CSV upload workflow...")
        print(f"Using project_id={self.project_id}")
        print(f"Using run_token={current_run_token}")
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
        
        # Delete all existing prospects for this project_id (wipe-and-replace behavior)
        print(f"Deleting existing prospects for project_id={self.project_id}...")
        try:
            delete_response = (
                self.supabase.table('prospects')
                .delete()
                .eq('project_id', self.project_id)
                .execute()
            )
            print(f"Deleted existing prospects for project_id={self.project_id}")
        except Exception as e:
            print(f"Error deleting existing prospects: {str(e)}")
            # Continue anyway - may be schema not ready yet
        
        print("-" * 50)
        
        # Fetch existing websites (should be empty after delete, but check for safety)
        print("Checking for existing websites in database...")
        existing_websites = self.get_existing_websites()
        print(f"Found {len(existing_websites)} existing websites in database")
        
        print("-" * 50)
        
        # Upload new rows
        print("Uploading new rows...")
        inserted_count, skipped_blank_count, skipped_duplicate_count, failed_count, exa_overwrites_applied, exa_values_used_for_new_rows = self.upload_rows(rows, existing_websites, current_run_token)
        
        print("-" * 50)
        print("Upload Summary:")
        print(f"  Rows read: {rows_read}")
        print(f"  Rows skipped (blank website): {skipped_blank_count}")
        print(f"  Rows skipped (duplicates): {skipped_duplicate_count}")
        print(f"  Rows inserted: {inserted_count}")
        print(f"  Rows failed: {failed_count}")
        print(f"  exa_overwrites_applied: {exa_overwrites_applied}")
        print(f"  exa_values_used_for_new_rows: {exa_values_used_for_new_rows}")
        print(f"Uploaded project={self.project_id} run_token={current_run_token} rows={inserted_count}")
        print("Upload workflow completed!")


def run(project_id: str, csv_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Run the upload workflow with provided CSV rows.
    
    Args:
        project_id: Project ID to scope all operations
        csv_rows: List of CSV row dictionaries (from DictReader)
        
    Returns:
        Dictionary with upload results including status, project_id, and row counts
    """
    # Generate run token at the start
    current_run_token = str(uuid.uuid4())
    
    print("Starting CSV upload workflow...")
    print(f"Using project_id={project_id}")
    print(f"Using run_token={current_run_token}")
    print("-" * 50)
    
    try:
        uploader = CSVUploader(project_id)
        
        # Test connection
        print("Testing Supabase connection...")
        if not uploader.test_connection():
            raise Exception("Cannot proceed without a valid Supabase connection.")
        
        print("-" * 50)
        
        # Extract and upload prompts FIRST (before prospect upload)
        print("Extracting prompts from CSV rows...")
        try:
            prompts_deduplicated, prompts_found_raw = uploader.extract_prompts_from_rows(csv_rows)
            prompts_inserted = len(prompts_deduplicated)
            
            print(f"prompts_found_raw: {prompts_found_raw}")
            print(f"prompts_inserted: {prompts_inserted}")
            
            # If zero prompts found, log error and raise
            if prompts_inserted == 0:
                raise Exception("Zero prompts found in CSV. Aborting to prevent accidental deletion of prompts table.")
            
            # Upload prompts
            print("Uploading prompts to public.prompts...")
            uploader.upload_prompts(prompts_deduplicated)
            print(f"Deleted existing prompts and inserted {prompts_inserted} new prompts (with run_if).")
            
        except Exception as e:
            raise Exception(f"Failed to upload prompts: {str(e)}")
        
        print("-" * 50)
        
        # Parse rows from CSV for prospects
        print("Parsing CSV rows for prospects...")
        try:
            rows = uploader.parse_csv_rows(csv_rows)
            rows_read = len(rows)
            print(f"Found {rows_read} rows in CSV")
        except Exception as e:
            raise Exception(f"Error parsing CSV rows: {str(e)}")
        
        if not rows:
            print("No rows found in CSV rows.")
            return {
                "status": "ok",
                "project_id": project_id,
                "rows": 0,
                "inserted": 0,
                "skipped_blank": 0,
                "skipped_duplicate": 0,
                "failed": 0,
                "exa_overwrites_applied": 0,
                "exa_values_used_for_new_rows": 0
            }
        
        print("-" * 50)
        
        # Delete all existing prospects for this project_id (wipe-and-replace behavior)
        print(f"Deleting existing prospects for project_id={project_id}...")
        try:
            delete_response = (
                uploader.supabase.table('prospects')
                .delete()
                .eq('project_id', project_id)
                .execute()
            )
            print(f"Deleted existing prospects for project_id={project_id}")
        except Exception as e:
            print(f"Error deleting existing prospects: {str(e)}")
            # Continue anyway - may be schema not ready yet
        
        print("-" * 50)
        
        # Fetch existing websites (should be empty after delete, but check for safety)
        print("Checking for existing websites in database...")
        existing_websites = uploader.get_existing_websites()
        print(f"Found {len(existing_websites)} existing websites in database")
        
        print("-" * 50)
        
        # Upload new rows
        print("Uploading new rows...")
        inserted_count, skipped_blank_count, skipped_duplicate_count, failed_count, exa_overwrites_applied, exa_values_used_for_new_rows = uploader.upload_rows(rows, existing_websites, current_run_token)
        
        print("-" * 50)
        print("Upload Summary:")
        print(f"  Rows read: {rows_read}")
        print(f"  Rows skipped (blank website): {skipped_blank_count}")
        print(f"  Rows skipped (duplicates): {skipped_duplicate_count}")
        print(f"  Rows inserted: {inserted_count}")
        print(f"  Rows failed: {failed_count}")
        print(f"  exa_overwrites_applied: {exa_overwrites_applied}")
        print(f"  exa_values_used_for_new_rows: {exa_values_used_for_new_rows}")
        print(f"Uploaded project={project_id} run_token={current_run_token} rows={inserted_count}")
        print("Upload workflow completed!")
        
        return {
            "status": "ok",
            "project_id": project_id,
            "rows": rows_read,
            "inserted": inserted_count,
            "skipped_blank": skipped_blank_count,
            "skipped_duplicate": skipped_duplicate_count,
            "failed": failed_count,
            "exa_overwrites_applied": exa_overwrites_applied,
            "exa_values_used_for_new_rows": exa_values_used_for_new_rows
        }
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise


def main():
    """Entry point for the script."""
    # Get project_id from command line or environment variable
    project_id = None
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
    else:
        project_id = os.getenv("PROJECT_ID")
    
    if not project_id:
        print("ERROR: project_id is required.")
        print("Usage: python upload_csv.py <project_id>")
        print("   OR: Set PROJECT_ID environment variable")
        sys.exit(1)
    
    try:
        uploader = CSVUploader(project_id)
        uploader.run()
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
