"""
XLSX ingestion module for uploading Excel files to Supabase run_sheet_rows table.

This module handles:
- Loading Excel files (XLSX) using pandas
- Wiping existing rows for a project_id
- Iterating through all sheets in the Excel file
- Converting dataframes to JSON and inserting into run_sheet_rows
"""

import json
import io
from typing import Dict, Any
from supabase import Client
import pandas as pd


def ingest_project_file(project_id: str, file_bytes: bytes, supabase_client: Client) -> Dict[str, Any]:
    """
    Ingest an Excel file into Supabase run_sheet_rows table.
    
    This function:
    1. Loads the Excel file into a pandas ExcelFile
    2. Wipes existing rows for the project_id
    3. Iterates through every sheet in the Excel file
    4. Converts each dataframe to list of dicts (JSON)
    5. Batch inserts into run_sheet_rows with run_id=NULL
    
    Args:
        project_id: Project ID to scope all operations
        file_bytes: Raw bytes of the Excel file
        supabase_client: Supabase client instance
        
    Returns:
        Summary dict with keys:
        - tabs: List of sheet names processed
        - rows: Total number of rows inserted
    """
    # Load bytes into Pandas ExcelFile
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    
    # WIPE: Delete all existing rows for this project_id
    try:
        supabase_client.table("run_sheet_rows").delete().eq("project_id", project_id).execute()
    except Exception as e:
        raise Exception(f"Failed to wipe existing rows for project_id={project_id}: {str(e)}")
    
    tabs = []
    total_rows = 0
    
    # LOOP: Iterate through every sheet in the Excel file
    for sheet_name in excel_file.sheet_names:
        tabs.append(sheet_name)
        
        # Read the sheet into a dataframe
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Skip empty sheets
        if df.empty:
            continue
        
        # TRANSFORM: Convert dataframe to list of dicts (JSON)
        # Replace NaN with empty strings and convert to dict
        df = df.fillna("")
        rows_dict = df.to_dict(orient='records')
        
        # Prepare rows for batch insert
        rows_to_insert = []
        for row_idx, row_dict in enumerate(rows_dict):
            # row_index is 0-based for data rows (first data row = 0)
            rows_to_insert.append({
                "project_id": project_id,
                "sheet_name": sheet_name,
                "row_index": row_idx,
                "row_json": json.dumps(row_dict),
                "run_id": None  # NULL as specified
            })
        
        # INSERT: Batch insert into run_sheet_rows
        if rows_to_insert:
            try:
                # Insert in batches to avoid payload size limits
                batch_size = 1000
                for i in range(0, len(rows_to_insert), batch_size):
                    batch = rows_to_insert[i:i + batch_size]
                    supabase_client.table("run_sheet_rows").insert(batch).execute()
                
                total_rows += len(rows_to_insert)
            except Exception as e:
                raise Exception(f"Failed to insert rows for sheet '{sheet_name}': {str(e)}")
    
    # RETURN: Summary dict
    return {
        "tabs": tabs,
        "rows": total_rows
    }
