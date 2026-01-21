"""
Snapshot cleanup script for Lead Pipeline V3.

This script enforces retention for Supabase snapshot tables:
- run_sheet_rows
- run_sheet_headers
- run_sheets

It keeps only the latest N runs worth of snapshot data per project_id,
deleting older snapshot records to prevent storage growth.

Usage:
    python3 scripts/cleanup_snapshots.py

Environment Variables:
    SNAPSHOT_RETENTION_RUNS: Number of latest runs to keep per project (default: 3)
    SUPABASE_URL: Supabase project URL (required)
    SUPABASE_KEY: Supabase service role key (required)
"""

import os
import sys
from typing import List, Set, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SNAPSHOT_RETENTION_RUNS = int(os.getenv("SNAPSHOT_RETENTION_RUNS", "3"))

# Chunk size for deletion operations (to avoid overly large IN clauses)
DELETE_CHUNK_SIZE = 50


def log(message: str) -> None:
    """Print a log message with [CLEANUP] prefix."""
    print(f"[CLEANUP] {message}")


def get_supabase_client() -> Client:
    """
    Create and return a Supabase client.
    
    Returns:
        Supabase client instance
        
    Raises:
        SystemExit: If Supabase configuration is missing
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("ERROR: SUPABASE_URL or SUPABASE_KEY environment variables are not set")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_distinct_project_ids(supabase: Client) -> List[str]:
    """
    Get all distinct project_id values from the runs table.
    
    Args:
        supabase: Supabase client instance
        
    Returns:
        List of distinct project_id values
        
    Raises:
        Exception: If query fails
    """
    try:
        # Use a select with distinct on project_id
        # Supabase doesn't have a direct distinct() method, so we'll fetch all and deduplicate
        response = supabase.table("runs").select("project_id").execute()
        
        if not response.data:
            return []
        
        # Extract unique project_ids
        project_ids = set()
        for row in response.data:
            project_id = row.get("project_id")
            if project_id:
                project_ids.add(project_id)
        
        return sorted(list(project_ids))
    except Exception as e:
        log(f"ERROR: Failed to fetch distinct project_ids: {str(e)}")
        raise


def get_latest_run_ids(supabase: Client, project_id: str, limit: int) -> List[int]:
    """
    Get the latest N run IDs for a project_id, ordered by created_at desc (or id desc).
    
    Args:
        supabase: Supabase client instance
        project_id: Project ID to filter by
        limit: Number of latest runs to keep
        
    Returns:
        List of run IDs (integers) to keep
    """
    try:
        # Try ordering by created_at first, fallback to id if created_at is unavailable
        response = (
            supabase.table("runs")
            .select("id")
            .eq("project_id", project_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        
        if not response.data:
            return []
        
        run_ids = [row["id"] for row in response.data if row.get("id") is not None]
        return run_ids
    except Exception as e:
        # If created_at ordering fails, try id desc
        try:
            log(f"  Warning: created_at ordering failed, trying id desc: {str(e)}")
            response = (
                supabase.table("runs")
                .select("id")
                .eq("project_id", project_id)
                .order("id", desc=True)
                .limit(limit)
                .execute()
            )
            
            if not response.data:
                return []
            
            run_ids = [row["id"] for row in response.data if row.get("id") is not None]
            return run_ids
        except Exception as e2:
            log(f"  ERROR: Failed to fetch latest run IDs: {str(e2)}")
            raise


def get_all_run_ids_for_project(supabase: Client, project_id: str) -> List[int]:
    """
    Get all run IDs for a project_id.
    
    Args:
        supabase: Supabase client instance
        project_id: Project ID to filter by
        
    Returns:
        List of all run IDs for the project
    """
    try:
        all_run_ids = []
        offset = 0
        batch_size = 1000
        
        while True:
            response = (
                supabase.table("runs")
                .select("id")
                .eq("project_id", project_id)
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            
            if not response.data:
                break
            
            for row in response.data:
                run_id = row.get("id")
                if run_id is not None:
                    all_run_ids.append(run_id)
            
            if len(response.data) < batch_size:
                break
            
            offset += batch_size
        
        return all_run_ids
    except Exception as e:
        log(f"  ERROR: Failed to fetch all run IDs: {str(e)}")
        raise


def delete_in_chunks(
    supabase: Client,
    table_name: str,
    run_ids: List[int],
    project_id: str
) -> int:
    """
    Delete rows from a table in chunks to avoid overly large IN clauses.
    
    Args:
        supabase: Supabase client instance
        table_name: Name of the table to delete from
        run_ids: List of run IDs to delete
        project_id: Project ID (for logging)
        
    Returns:
        Total number of rows deleted (approximate, based on response if available)
    """
    if not run_ids:
        return 0
    
    total_deleted = 0
    
    # Process in chunks
    for i in range(0, len(run_ids), DELETE_CHUNK_SIZE):
        chunk = run_ids[i:i + DELETE_CHUNK_SIZE]
        
        try:
            # Delete rows matching the run_ids in this chunk
            # Filter by both project_id and run_id for safety
            # Supabase delete() may return deleted rows in response.data
            response = (
                supabase.table(table_name)
                .delete()
                .eq("project_id", project_id)
                .in_("run_id", chunk)
                .execute()
            )
            
            # Count deleted rows if available in response
            # Note: Some Supabase configurations may not return deleted rows
            if response.data:
                deleted_count = len(response.data)
            else:
                # If response.data is None/empty, we can't count exactly
                # Log that deletion was attempted but count is unknown
                deleted_count = None
            
            if deleted_count is not None:
                total_deleted += deleted_count
                log(f"    Deleted {deleted_count} rows from {table_name} (chunk {i//DELETE_CHUNK_SIZE + 1})")
            else:
                log(f"    Deleted chunk from {table_name} (chunk {i//DELETE_CHUNK_SIZE + 1}, count unavailable)")
        except Exception as e:
            log(f"    ERROR: Failed to delete from {table_name} for chunk: {str(e)}")
            # Continue with next chunk instead of failing completely
            continue
    
    return total_deleted


def cleanup_project_snapshots(supabase: Client, project_id: str, retention_runs: int) -> Dict[str, Any]:
    """
    Clean up snapshot data for a single project_id.
    
    Args:
        supabase: Supabase client instance
        project_id: Project ID to clean up
        retention_runs: Number of latest runs to keep
        
    Returns:
        Dictionary with cleanup statistics:
        - runs_cleaned: Number of runs that had data deleted
        - rows_deleted: Total rows deleted from run_sheet_rows
        - headers_deleted: Total rows deleted from run_sheet_headers
        - sheets_deleted: Total rows deleted from run_sheets
        - success: Boolean indicating if cleanup succeeded
    """
    stats = {
        "runs_cleaned": 0,
        "rows_deleted": 0,
        "headers_deleted": 0,
        "sheets_deleted": 0,
        "success": False
    }
    
    try:
        log(f"Processing project_id: {project_id}")
        
        # Step 1: Get latest N run IDs to keep
        keep_run_ids = get_latest_run_ids(supabase, project_id, retention_runs)
        keep_set = set(keep_run_ids)
        
        log(f"  Keeping latest {len(keep_set)} run(s): {sorted(keep_set)}")
        
        # Step 2: Get all run IDs for this project
        all_run_ids = get_all_run_ids_for_project(supabase, project_id)
        
        # Step 3: Find run IDs to delete (not in keep set)
        delete_run_ids = [rid for rid in all_run_ids if rid not in keep_set]
        
        if not delete_run_ids:
            log(f"  No runs to clean up (all {len(all_run_ids)} runs are within retention)")
            stats["success"] = True
            return stats
        
        log(f"  Found {len(delete_run_ids)} run(s) to clean up: {sorted(delete_run_ids)}")
        stats["runs_cleaned"] = len(delete_run_ids)
        
        # Step 4: Delete snapshot data in correct order
        # Order: run_sheet_rows -> run_sheet_headers -> run_sheets
        
        log(f"  Deleting from run_sheet_rows...")
        rows_deleted = delete_in_chunks(supabase, "run_sheet_rows", delete_run_ids, project_id)
        stats["rows_deleted"] = rows_deleted
        
        log(f"  Deleting from run_sheet_headers...")
        headers_deleted = delete_in_chunks(supabase, "run_sheet_headers", delete_run_ids, project_id)
        stats["headers_deleted"] = headers_deleted
        
        log(f"  Deleting from run_sheets...")
        sheets_deleted = delete_in_chunks(supabase, "run_sheets", delete_run_ids, project_id)
        stats["sheets_deleted"] = sheets_deleted
        
        log(f"  Completed cleanup for {project_id}: {rows_deleted} rows, {headers_deleted} headers, {sheets_deleted} sheets")
        stats["success"] = True
        
    except Exception as e:
        log(f"  ERROR: Failed to clean up project_id {project_id}: {str(e)}")
        stats["success"] = False
    
    return stats


def main() -> None:
    """Main entry point for the cleanup script."""
    log("Starting snapshot cleanup script")
    log(f"Retention setting: Keep latest {SNAPSHOT_RETENTION_RUNS} runs per project")
    
    # Initialize Supabase client
    try:
        supabase = get_supabase_client()
        log("Connected to Supabase")
    except Exception as e:
        log(f"FATAL ERROR: Failed to initialize Supabase client: {str(e)}")
        sys.exit(1)
    
    # Get all distinct project_ids
    try:
        project_ids = get_distinct_project_ids(supabase)
        log(f"Found {len(project_ids)} project(s) to process")
    except Exception as e:
        log(f"FATAL ERROR: Failed to fetch project_ids: {str(e)}")
        sys.exit(1)
    
    if not project_ids:
        log("No projects found, exiting")
        return
    
    # Process each project
    total_stats = {
        "projects_processed": 0,
        "projects_succeeded": 0,
        "projects_failed": 0,
        "total_runs_cleaned": 0,
        "total_rows_deleted": 0,
        "total_headers_deleted": 0,
        "total_sheets_deleted": 0
    }
    
    for project_id in project_ids:
        stats = cleanup_project_snapshots(supabase, project_id, SNAPSHOT_RETENTION_RUNS)
        
        total_stats["projects_processed"] += 1
        if stats["success"]:
            total_stats["projects_succeeded"] += 1
        else:
            total_stats["projects_failed"] += 1
        
        total_stats["total_runs_cleaned"] += stats["runs_cleaned"]
        total_stats["total_rows_deleted"] += stats["rows_deleted"]
        total_stats["total_headers_deleted"] += stats["headers_deleted"]
        total_stats["total_sheets_deleted"] += stats["sheets_deleted"]
    
    # Print summary
    log("")
    log("=" * 60)
    log("CLEANUP SUMMARY")
    log("=" * 60)
    log(f"Projects processed: {total_stats['projects_processed']}")
    log(f"Projects succeeded: {total_stats['projects_succeeded']}")
    log(f"Projects failed: {total_stats['projects_failed']}")
    log(f"Total runs cleaned: {total_stats['total_runs_cleaned']}")
    log(f"Total rows deleted (run_sheet_rows): {total_stats['total_rows_deleted']}")
    log(f"Total headers deleted (run_sheet_headers): {total_stats['total_headers_deleted']}")
    log(f"Total sheets deleted (run_sheets): {total_stats['total_sheets_deleted']}")
    log("=" * 60)
    
    # Exit with non-zero if there were failures
    if total_stats["projects_failed"] > 0:
        log("WARNING: Some projects failed to process")
        sys.exit(1)
    
    log("Cleanup completed successfully")


if __name__ == "__main__":
    main()
