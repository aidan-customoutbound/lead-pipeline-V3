"""
Bulk upload script for prospect websites.

This script reads websites from input.csv and uploads them to the Supabase prospects table,
skipping duplicates.
"""

import csv
import os
from typing import Set, List
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
INPUT_CSV = "input.csv"


class CSVUploader:
    """Handles CSV upload to Supabase prospects table."""
    
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
                
                # Normalize websites (lowercase, strip whitespace)
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
        Normalize website URL for comparison (remove protocol, lowercase, strip).
        
        Args:
            website: Website URL to normalize
            
        Returns:
            Normalized website string
        """
        if not website:
            return ""
        
        website = website.strip().lower()
        
        # Remove protocol
        for protocol in ['https://', 'http://', 'www.']:
            if website.startswith(protocol):
                website = website[len(protocol):]
        
        # Remove trailing slash
        website = website.rstrip('/')
        
        return website
    
    def read_csv_websites(self, csv_path: str) -> List[str]:
        """
        Read websites from CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of website strings
        """
        websites = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check if 'website' column exists
                if 'website' not in reader.fieldnames:
                    raise ValueError(
                        f"CSV file must have a 'website' column. "
                        f"Found columns: {', '.join(reader.fieldnames or [])}"
                    )
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                    website = row.get('website', '').strip()
                    if website:
                        websites.append(website)
                    elif website == '':
                        # Skip empty rows but don't error
                        continue
            
            return websites
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
    
    def upload_websites(self, websites: List[str], existing_websites: Set[str]) -> tuple[int, int]:
        """
        Upload new websites to Supabase, skipping duplicates.
        
        Args:
            websites: List of websites to upload
            existing_websites: Set of normalized existing websites
            
        Returns:
            Tuple of (uploaded_count, skipped_count)
        """
        uploaded_count = 0
        skipped_count = 0
        new_websites = []
        
        # Filter out duplicates
        for website in websites:
            normalized = self._normalize_website(website)
            
            if normalized in existing_websites:
                skipped_count += 1
            else:
                new_websites.append({
                    'website': website,
                    'status': 'new'
                })
                # Add to existing set to avoid duplicates within the same batch
                existing_websites.add(normalized)
        
        # Batch insert new websites
        if new_websites:
            try:
                # Insert in batches to avoid payload size limits
                batch_size = 100
                for i in range(0, len(new_websites), batch_size):
                    batch = new_websites[i:i + batch_size]
                    response = (
                        self.supabase.table('prospects')
                        .insert(batch)
                        .execute()
                    )
                    uploaded_count += len(batch)
                    print(f"Uploaded batch: {len(batch)} websites (total: {uploaded_count})")
            except Exception as e:
                print(f"Error uploading websites: {str(e)}")
                # Return partial count if some were uploaded before error
                return uploaded_count, skipped_count
        
        return uploaded_count, skipped_count
    
    def run(self) -> None:
        """Main upload workflow execution."""
        print("Starting CSV upload workflow...")
        print("-" * 50)
        
        # Test connection
        print("Testing Supabase connection...")
        if not self.test_connection():
            print("Cannot proceed without a valid Supabase connection.")
            return
        
        print("-" * 50)
        
        # Check if CSV file exists
        if not os.path.exists(INPUT_CSV):
            print(f"Error: {INPUT_CSV} not found in the current directory.")
            return
        
        # Read websites from CSV
        print(f"Reading websites from {INPUT_CSV}...")
        try:
            websites = self.read_csv_websites(INPUT_CSV)
            print(f"Found {len(websites)} websites in CSV")
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            return
        
        if not websites:
            print("No websites found in CSV file.")
            return
        
        print("-" * 50)
        
        # Fetch existing websites
        print("Checking for existing websites in database...")
        existing_websites = self.get_existing_websites()
        print(f"Found {len(existing_websites)} existing websites in database")
        
        print("-" * 50)
        
        # Upload new websites
        print("Uploading new websites...")
        uploaded_count, skipped_count = self.upload_websites(websites, existing_websites)
        
        print("-" * 50)
        print(f"Uploaded {uploaded_count} new websites. Skipped {skipped_count} duplicates.")
        print("Upload workflow completed!")


def main():
    """Entry point for the script."""
    try:
        uploader = CSVUploader()
        uploader.run()
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
