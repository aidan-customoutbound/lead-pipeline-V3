"""
Worker process that polls for queued runs and processes them.

This worker continuously polls the public.runs table for runs with status='queued',
claims them atomically, and executes the enrichment workflow.
"""

import asyncio
import builtins
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from api_server import get_supabase_client

import enrich_workflow
import recipe_workflow
from sheet_export import (
    get_sheets_service,
    get_sheet_id_for_project,
    read_tab_as_rows,
    write_rows_to_tab,
    update_master_statuses
)

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def log(message: str) -> None:
    """Print a log message with [worker] prefix."""
    print(f"[worker] {message}")


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


def is_run_active(supabase_client, run_id):
    """
    Returns True only if the current run's status is 'running'.
    Returns False for any other status (queued, completed, failed, superseded).
    """
    try:
        resp = (
            supabase_client.table("runs")
            .select("status")
            .eq("id", run_id)
            .single()
            .execute()
        )
        if not resp.data:
            return False
        return resp.data.get("status") == "running"
    except Exception:
        return False


def claim_next_run(supabase: Client) -> Optional[Dict[str, Any]]:
    """
    Atomically claim the oldest queued run.
    
    This function:
    1. Finds the oldest run with status='queued'
    2. Atomically updates it to status='running' and sets started_at
    3. Returns the full row (id + project_id + run_type) if successful
    4. Returns None if no queued runs are found
    
    Args:
        supabase: Supabase client instance
        
    Returns:
        Dictionary with run data (id, project_id, run_type) if a run was claimed, None otherwise
    """
    try:
        # Find the oldest queued run
        response = supabase.table("runs").select("id, project_id, run_type").eq("status", "queued").order("created_at", desc=False).limit(1).execute()
        
        if not response.data or len(response.data) == 0:
            return None
        
        run = response.data[0]
        run_id = run["id"]
        
        # Atomically update: only update if status is still 'queued'
        # This ensures we don't claim a run that another worker already claimed
        started_at = datetime.utcnow()
        update_response = supabase.table("runs").update({
            "status": "running",
            "started_at": started_at.isoformat()
        }).eq("id", run_id).eq("status", "queued").execute()
        
        # Check if the update actually affected a row
        if not update_response.data or len(update_response.data) == 0:
            # Another worker claimed it first, try again next iteration
            return None
        
        log(f"Claimed run {run_id} for project {run['project_id']}, run_type={run.get('run_type', 'enrichment')}")
        return run
        
    except Exception as e:
        log(f"Error claiming run: {str(e)}")
        return None


def process_run(run_row, supabase):
    """
    Process a claimed run by executing the appropriate workflow based on run_type.
    
    Args:
        run_row: Dictionary with run data (id, project_id, run_type)
        supabase: Supabase client instance
    """
    run_id = run_row["id"]
    project_id = run_row["project_id"]
    run_type = run_row.get("run_type") or "enrichment"
    
    log(f"processing run {run_id} for project {project_id}, run_type={run_type}")
    
    # Check if run is still active before starting
    if not is_run_active(supabase, run_id):
        log(f"Run {run_id} is no longer active, stopping processing")
        return
    
    # Branch based on run_type
    if run_type == "recipe":
        # Recipe run: read from Sheets, run recipe_workflow, write results back
        log(f"[worker] Processing recipe run for run_id={run_id}, project={project_id}")
        
        # Check if run is still active
        if not is_run_active(supabase, run_id):
            log(f"Run {run_id} is no longer active, stopping recipe processing")
            return
        
        # Create list to capture [RECIPE] debug logs
        debug_logs = []
        
        # Store original print function
        original_print = builtins.print
        
        # Create wrapper that intercepts [RECIPE] prints
        def print_wrapper(*args, **kwargs):
            # Convert args to string to check prefix (use sep from kwargs or default space)
            sep = kwargs.get('sep', ' ')
            message = sep.join(str(arg) for arg in args)
            if message.startswith("[RECIPE]"):
                debug_logs.append(message)
            else:
                # Pass through to original print
                original_print(*args, **kwargs)
        
        try:
            # Get Google Sheets service
            service = get_sheets_service()
            if not service:
                raise ValueError("Could not create Google Sheets service. Check GOOGLE_SA_JSON environment variable.")
            
            # Get sheet_id for this project
            # This will fall back to project_id if not found in prompts table (safe for recipe runs)
            sheet_id = get_sheet_id_for_project(project_id, supabase)
            if not sheet_id:
                # This should only happen if project_id itself is invalid/empty
                raise ValueError(f"Invalid project_id: {project_id}")
            
            log(f"[worker] Reading tabs from sheet_id={sheet_id}")
            
            # Read input tabs
            urls_rows = read_tab_as_rows(service, sheet_id, "URLs")
            contacts_rows = read_tab_as_rows(service, sheet_id, "Contacts")
            master_rows = read_tab_as_rows(service, sheet_id, "Master")
            
            log(f"[worker] Read {len(urls_rows)} URLs rows, {len(contacts_rows)} Contacts rows, {len(master_rows)} Master rows")
            
            # Check if run is still active before running recipe
            if not is_run_active(supabase, run_id):
                log(f"Run {run_id} is no longer active, stopping before running recipe")
                return
            
            # Define progress callback for real-time status updates
            def recipe_progress_callback(row_index: int, status: str) -> None:
                # 1) If run has been superseded, do nothing
                if not is_run_active(supabase, run_id):
                    return
                
                try:
                    # Use the existing update_master_statuses helper
                    update_master_statuses(
                        service,
                        sheet_id,
                        "Master",
                        [{"row_index": row_index, "status": status}],
                    )
                except Exception as e:
                    log(f"[recipe] Warning: failed to update Master status for row {row_index}: {e}")
            
            # Patch print to capture [RECIPE] logs
            builtins.print = print_wrapper
            
            try:
                # Run the recipe workflow
                log(f"[worker] Running recipe_workflow.run_recipe(...)")
                result = recipe_workflow.run_recipe(
                    project_id=project_id,
                    run_id=run_id,
                    urls_rows=urls_rows,
                    contacts_rows=contacts_rows,
                    master_rows=master_rows,
                    progress_callback=recipe_progress_callback
                )
            finally:
                # Always restore original print
                builtins.print = original_print
            
            # Check if run is still active after recipe execution
            if not is_run_active(supabase, run_id):
                log(f"Run {run_id} is no longer active, stopping before writing results")
                return
            
            # Store debug logs in error_message column (regardless of success/failure)
            debug_logs_text = "\n".join(debug_logs) if debug_logs else ""
            
            # Handle recipe result
            if not result.get("ok", False):
                # Recipe failed - mark run as failed, do NOT write to Sheets
                errors = result.get("errors", [])
                error_message = "; ".join(errors) if errors else "Recipe execution failed"
                error_message = error_message[:500]  # Truncate to first 500 chars
                
                # Combine error message with debug logs
                if debug_logs_text:
                    combined_message = f"{error_message}\n\n[DEBUG LOGS]\n{debug_logs_text}"
                    # Truncate combined message if too long (keep error message priority)
                    if len(combined_message) > 5000:
                        combined_message = combined_message[:5000]
                    error_message = combined_message
                
                finished_at = datetime.utcnow()
                supabase.table("runs").update({
                    "status": "failed",
                    "finished_at": finished_at.isoformat(),
                    "error_message": error_message
                }).eq("id", run_id).eq("status", "running").execute()
                
                log(f"Failed recipe run {run_id} for project {project_id}: {error_message}")
                return
            
            # Recipe succeeded - write results to Sheets
            log(f"[worker] Recipe succeeded, writing results to Sheets")
            
            # Write URLs output
            urls_output = result.get("urls_output")
            if urls_output is not None:
                log(f"[worker] Writing {len(urls_output)} rows to 'URLs output' tab")
                write_rows_to_tab(service, sheet_id, "URLs output", urls_output)
            
            # Write Contacts output
            contacts_output = result.get("contacts_output")
            if contacts_output is not None:
                log(f"[worker] Writing {len(contacts_output)} rows to 'Contacts output' tab")
                write_rows_to_tab(service, sheet_id, "Contacts output", contacts_output)
            
            # Update Master statuses
            master_status_updates = result.get("master_status_updates")
            if master_status_updates:
                log(f"[worker] Updating {len(master_status_updates)} Master status rows")
                update_master_statuses(service, sheet_id, "Master", master_status_updates)
            
            # Check if run is still active before marking as completed
            if not is_run_active(supabase, run_id):
                log(f"Run {run_id} is no longer active, stopping before marking completed")
                return
            
            # Mark run as completed (store debug logs in error_message even on success)
            finished_at = datetime.utcnow()
            update_data = {
                "status": "completed",
                "finished_at": finished_at.isoformat()
            }
            if debug_logs_text:
                # Truncate if too long
                update_data["error_message"] = debug_logs_text[:5000] if len(debug_logs_text) > 5000 else debug_logs_text
            
            supabase.table("runs").update(update_data).eq("id", run_id).eq("status", "running").execute()
            
            log(f"Completed recipe run {run_id} for project {project_id}")
            
        except Exception as e:
            # Restore original print if exception occurred before finally block
            builtins.print = original_print
            
            # Check if run is still active before marking as failed
            if not is_run_active(supabase, run_id):
                log(f"Run {run_id} is no longer active, stopping before marking failed")
                return
            
            # Update run to failed (only if still in 'running' status to prevent superseded runs from updating)
            finished_at = datetime.utcnow()
            error_message = str(e)[:500]  # Truncate to first 500 chars
            
            # Combine error message with debug logs if available
            debug_logs_text = "\n".join(debug_logs) if debug_logs else ""
            if debug_logs_text:
                combined_message = f"{error_message}\n\n[DEBUG LOGS]\n{debug_logs_text}"
                # Truncate combined message if too long (keep error message priority)
                if len(combined_message) > 5000:
                    combined_message = combined_message[:5000]
                error_message = combined_message
            
            try:
                supabase.table("runs").update({
                    "status": "failed",
                    "finished_at": finished_at.isoformat(),
                    "error_message": error_message
                }).eq("id", run_id).eq("status", "running").execute()
            except Exception as update_error:
                log(f"Error updating run {run_id} to failed: {str(update_error)}")
            
            log(f"Failed recipe run {run_id} for project {project_id}: {error_message}")
    else:
        # Enrichment run (default): use existing enrichment workflow
        try:
            # Run the enrichment workflow (pass run_id for active checks)
            asyncio.run(enrich_workflow.run(project_id, run_id))
            
            # Check if run is still active before marking as completed
            if not is_run_active(supabase, run_id):
                log(f"Run {run_id} is no longer active, stopping before marking completed")
                return
            
            # Update run to completed (only if still in 'running' status to prevent superseded runs from updating)
            finished_at = datetime.utcnow()
            supabase.table("runs").update({
                "status": "completed",
                "finished_at": finished_at.isoformat()
            }).eq("id", run_id).eq("status", "running").execute()
            
            log(f"Completed run {run_id} for project {project_id}")
            
        except Exception as e:
            # Check if run is still active before marking as failed
            if not is_run_active(supabase, run_id):
                log(f"Run {run_id} is no longer active, stopping before marking failed")
                return
            
            # Update run to failed (only if still in 'running' status to prevent superseded runs from updating)
            finished_at = datetime.utcnow()
            error_message = str(e)[:500]  # Truncate to first 500 chars
            
            try:
                supabase.table("runs").update({
                    "status": "failed",
                    "finished_at": finished_at.isoformat(),
                    "error_message": error_message
                }).eq("id", run_id).eq("status", "running").execute()
            except Exception as update_error:
                log(f"Error updating run {run_id} to failed: {str(update_error)}")
            
            log(f"Failed run {run_id} for project {project_id}: {error_message}")
            # Don't re-raise - let the worker continue processing other runs


async def main():
    """Main worker loop that continuously polls for queued runs."""
    log("starting")
    
    # Initialize Supabase client
    supabase = get_supabase_client()
    
    # Main polling loop
    while True:
        try:
            # Try to claim the next queued run
            run = claim_next_run(supabase)
            
            if run is None:
                log("no queued runs, sleeping")
                await asyncio.sleep(5)
                continue
            
            # Process the claimed run
            process_run(run, supabase)
            
        except KeyboardInterrupt:
            log("interrupted by user, shutting down")
            break
        except Exception as e:
            log(f"Fatal error in main loop: {str(e)}")
            # Let it crash so Render restarts it
            raise


if __name__ == "__main__":
    print("[worker] Starting worker...")
    supabase = get_supabase_client()
    while True:
        try:
            run_row = claim_next_run(supabase)
            if not run_row:
                print("[worker] No queued runs... sleeping 5s")
                time.sleep(5)
                continue

            print(f"[worker] Processing run_id={run_row['id']}")
            process_run(run_row, supabase)

        except Exception as e:
            print(f"[worker] Fatal error in worker loop: {e}")
            time.sleep(5)


