"""
Google Sheets export module for writing enrichment results back to Google Sheets.

This module handles exporting completed prospect data from Supabase to a Google Sheet.
"""

import json
import os
from typing import List, Any, Dict
from dotenv import load_dotenv
from supabase import create_client, Client
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_SA_JSON = os.getenv("GOOGLE_SA_JSON")


def export_results_to_google_sheets(project_id: str) -> None:
    """
    Export all prospects for a project from Supabase to Google Sheets.
    
    Steps:
    1. Fetch sheet_id from prompts table for this project_id
    2. Read GOOGLE_SA_JSON from env and build a Sheets API client
    3. Query Supabase: SELECT * FROM prospects WHERE project_id=<id> ORDER BY id ASC
    4. Build a 2D list with first row = column names, following rows = values
    5. Clear the 'output' tab fully using spreadsheets.values.clear()
    6. Write the data starting at A1 using spreadsheets.values.update(..., valueInputOption="RAW")
    
    Args:
        project_id: Project ID to export results for
        
    Raises:
        Exception: If Sheets API fails (logged but not re-raised to avoid crashing worker)
    """
    try:
        # Validate required environment variables
        if not GOOGLE_SA_JSON:
            print("  [SHEET EXPORT] GOOGLE_SA_JSON not set, skipping export")
            return
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("  [SHEET EXPORT] Supabase credentials not set, skipping export")
            return
        
        # Step 1: Fetch sheet_id from prompts table
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        print(f"  [SHEET EXPORT] Fetching sheet_id for project_id={project_id}...")
        sheet_id_response = (
            supabase.table('prompts')
            .select('sheet_id')
            .eq('project_id', project_id)
            .eq('is_active', True)
            .limit(1)
            .execute()
        )
        
        sheet_id = None
        if sheet_id_response.data and len(sheet_id_response.data) > 0:
            sheet_id = sheet_id_response.data[0].get('sheet_id')
        
        if not sheet_id or not sheet_id.strip():
            print(f"  [SHEET EXPORT] No sheet_id found for project_id={project_id}, skipping export")
            return
        
        print(f"  [SHEET EXPORT] Using sheet_id={sheet_id}")
        print("  [SHEET EXPORT] Starting export to Google Sheets...")
        
        # Step 2: Build Google Sheets API client
        sa_credentials = json.loads(GOOGLE_SA_JSON)
        credentials = service_account.Credentials.from_service_account_info(
            sa_credentials,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        service = build('sheets', 'v4', credentials=credentials)
        sheets = service.spreadsheets()
        
        # Step 3: Query Supabase for all prospects
        print(f"  [SHEET EXPORT] Fetching prospects for project_id={project_id}...")
        
        # Fetch all rows with pagination
        all_prospects = []
        offset = 0
        fetch_batch_size = 1000
        
        while True:
            response = (
                supabase.table('prospects')
                .select('*')
                .eq('project_id', project_id)
                .order('id', desc=False)
                .range(offset, offset + fetch_batch_size - 1)
                .execute()
            )
            
            if not response.data:
                break
            
            all_prospects.extend(response.data)
            
            if len(response.data) < fetch_batch_size:
                break
            
            offset += fetch_batch_size
        
        row_count = len(all_prospects)
        print(f"  [SHEET EXPORT] Fetched {row_count} prospects from Supabase")
        
        # Step 4: Build 2D list (first row = headers, following rows = values)
        if row_count == 0:
            # If no rows, clear the tab and exit cleanly
            print("  [SHEET EXPORT] No rows to export, clearing output tab...")
            sheets.values().clear(
                spreadsheetId=sheet_id,
                range='output!A:ZZ'
            ).execute()
            print(f"  [SHEET EXPORT] Export complete for {project_id}: 0 rows → {sheet_id}/output")
            return
        
        # Get column names from first row
        first_row = all_prospects[0]
        column_names = list(first_row.keys())
        
        # Build 2D list: headers first, then data rows
        data_rows = [column_names]  # First row is headers
        
        for prospect in all_prospects:
            row_values = []
            for col_name in column_names:
                value = prospect.get(col_name)
                # Convert None to empty string, other values to string
                if value is None:
                    row_values.append('')
                else:
                    row_values.append(str(value))
            data_rows.append(row_values)
        
        print(f"  [SHEET EXPORT] Built 2D list: {len(data_rows)} rows (1 header + {row_count} data)")
        
        # Step 5: Clear the 'output' tab fully
        print("  [SHEET EXPORT] Clearing output tab...")
        sheets.values().clear(
            spreadsheetId=sheet_id,
            range='output!A:ZZ'
        ).execute()
        
        # Step 6: Write data starting at A1
        print("  [SHEET EXPORT] Writing data to output tab starting at A1...")
        sheets.values().update(
            spreadsheetId=sheet_id,
            range='output!A1',
            valueInputOption='RAW',
            body={'values': data_rows}
        ).execute()
        
        print(f"  [SHEET EXPORT] Export complete for {project_id}: {row_count} rows → {sheet_id}/output")
        
    except Exception as e:
        # Log error but DO NOT crash the worker
        print(f"  [SHEET EXPORT] Error exporting to Google Sheets: {str(e)}")
        print(f"  [SHEET EXPORT] Export failed, but worker continues normally")

