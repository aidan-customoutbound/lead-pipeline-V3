"""
Worker process that polls for queued runs and processes them.

This worker continuously polls the public.runs table for runs with status='queued',
claims them atomically, and executes the recipe workflow.
"""

import asyncio
import builtins
import os
import sys
import socket
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client, Client
from api_server import get_supabase_client

import recipe_workflow
from sheet_export import (
    get_sheets_service,
    get_sheet_id_for_project,
    read_tab_as_rows,
    write_rows_to_tab,
    update_master_statuses
)
from services.snapshot_ingest import ingest_spreadsheet_to_supabase
from services.project_lock import acquire_lock, heartbeat_lock, release_lock

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Snapshot ingestion feature flag
SNAPSHOT_INGEST_ENABLED = os.getenv("SNAPSHOT_INGEST_ENABLED", "false").lower() == "true"

# Lock configuration
RUN_LOCK_TTL_SECONDS = int(os.getenv("RUN_LOCK_TTL_SECONDS", "900"))
RUN_LOCK_BUSY_REQUEUE_SLEEP_SECONDS = int(os.getenv("RUN_LOCK_BUSY_REQUEUE_SLEEP_SECONDS", "5"))

# Generate stable worker_id
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


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


def _insert_task_stats(supabase_client, run_id: int, project_id: str, ai_task_stats: List[Dict[str, Any]]) -> None:
    """
    Insert per-task AI stats into run_task_stats table.
    
    This function is guarded with error handling so it won't crash if the table doesn't exist yet.
    
    Args:
        supabase_client: Supabase client instance
        run_id: Run ID
        project_id: Project ID
        ai_task_stats: List of task stat dicts with keys: task_index, task_name, calls, cost_credits, prompt_tokens, completion_tokens
    """
    if not ai_task_stats:
        return
    
    try:
        for task_stat in ai_task_stats:
            supabase_client.table("run_task_stats").insert({
                "run_id": run_id,
                "project_id": project_id,
                "task_index": task_stat.get("task_index"),
                "task_name": task_stat.get("task_name", ""),
                "ai_calls": task_stat.get("calls", 0),
                "ai_total_cost_credits": task_stat.get("cost_credits", 0.0),
                "ai_total_prompt_tokens": task_stat.get("prompt_tokens", 0),
                "ai_total_completion_tokens": task_stat.get("completion_tokens", 0)
            }).execute()
    except Exception as e:
        # Log error but don't crash - table may not exist yet
        log(f"Warning: Could not insert task stats for run {run_id}: {str(e)}")


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
        update_data = {
            "status": "running",
            "started_at": started_at.isoformat()
        }
        # Set claimed_by if the column exists
        if WORKER_ID:
            update_data["claimed_by"] = WORKER_ID
        
        update_response = supabase.table("runs").update(update_data).eq("id", run_id).eq("status", "queued").execute()
        
        # Check if the update actually affected a row
        if not update_response.data or len(update_response.data) == 0:
            # Another worker claimed it first, try again next iteration
            return None
        
        log(f"Claimed run {run_id} for project {run['project_id']}, run_type={run.get('run_type', 'recipe')}, worker={WORKER_ID}")
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
    run_type = run_row.get("run_type") or "recipe"
    
    log(f"processing run {run_id} for project {project_id}, run_type={run_type}")
    
    # Check if run is still active before starting
    if not is_run_active(supabase, run_id):
        log(f"Run {run_id} is no longer active, stopping processing")
        return
    
    # Acquire lock immediately after claiming (before snapshot ingestion)
    lock_acquired = False
    try:
        try:
            lock_acquired = acquire_lock(supabase, project_id, run_id, WORKER_ID, RUN_LOCK_TTL_SECONDS)
        except Exception as lock_error:
            # RPC call failed - mark run as failed with clear error message
            error_message = f"Failed to acquire run lock: {str(lock_error)}"
            log(f"ERROR: {error_message}")
            finished_at = datetime.utcnow()
            try:
                supabase.table("runs").update({
                    "status": "failed",
                    "finished_at": finished_at.isoformat(),
                    "error_message": error_message[:500]
                }).eq("id", run_id).eq("status", "running").execute()
            except Exception as update_error:
                log(f"Error updating run {run_id} to failed: {str(update_error)}")
            return
        
        if not lock_acquired:
            # Lock is busy - requeue the run
            log(f"Lock busy for project {project_id}, run {run_id}; requeuing")
            try:
                supabase.table("runs").update({
                    "status": "queued",
                    "error_message": "Run lock busy; requeued"
                }).eq("id", run_id).eq("status", "running").execute()
            except Exception as requeue_error:
                log(f"Error requeuing run {run_id}: {str(requeue_error)}")
                # If requeue fails, mark as failed
                try:
                    supabase.table("runs").update({
                        "status": "failed",
                        "error_message": f"Lock busy and requeue failed: {str(requeue_error)}"
                    }).eq("id", run_id).eq("status", "running").execute()
                except:
                    pass
            # Sleep briefly before returning to prevent tight thrash
            time.sleep(RUN_LOCK_BUSY_REQUEUE_SLEEP_SECONDS)
            return
        
        # Lock acquired successfully - proceed with processing
        # Wrap entire processing in try/finally to ensure lock is released
        try:
            # Only support recipe runs - enrichment workflow has been deprecated
            if run_type == "recipe" or not run_type or run_type.strip() == "":
                # Recipe run: read from Sheets, run recipe_workflow, write results back
                log(f"[worker] Processing recipe run for run_id={run_id}, project={project_id}")
                
            # Only support recipe runs - enrichment workflow has been deprecated
            if run_type == "recipe" or not run_type or run_type.strip() == "":
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
                    
                    # For recipe runs, project_id IS the Google Sheet ID
                    # Do NOT use get_sheet_id_for_project or read from prompts table
                    sheet_id = project_id
                    if not sheet_id:
                        raise ValueError(f"Invalid project_id (sheet_id): {project_id}")
                    
                    log(f"[worker] [RECIPE] Using sheet_id={sheet_id} (from project_id, not prompts table)")
                    
                    # Snapshot ingestion phase (if enabled)
                    if SNAPSHOT_INGEST_ENABLED:
                        log(f"[worker] [SNAPSHOT] Snapshot ingestion enabled, starting ingestion phase")
                        
                        # Check if run is still active before ingestion
                        if not is_run_active(supabase, run_id):
                            log(f"Run {run_id} is no longer active, stopping before ingestion")
                            return
                        
                        # Heartbeat before ingestion begins
                        try:
                            heartbeat_lock(supabase, project_id, run_id, WORKER_ID, RUN_LOCK_TTL_SECONDS)
                        except Exception as heartbeat_error:
                            log(f"Warning: Heartbeat failed before ingestion: {str(heartbeat_error)}")
                            # Continue anyway - lock may still be valid
                        
                        try:
                            # Define callback to check if run is still active
                            def is_run_active_callback() -> bool:
                                return is_run_active(supabase, run_id)
                            
                            # Define heartbeat callback for ingestion batches
                            def lock_heartbeat_callback() -> None:
                                try:
                                    heartbeat_lock(supabase, project_id, run_id, WORKER_ID, RUN_LOCK_TTL_SECONDS)
                                except Exception as heartbeat_error:
                                    # Raise to abort ingestion if heartbeat fails
                                    raise Exception(f"Lock heartbeat failed: {str(heartbeat_error)}")
                            
                            # Perform snapshot ingestion
                            ingest_spreadsheet_to_supabase(
                                project_id=project_id,
                                run_id=run_id,
                                sheets_service=service,
                                supabase=supabase,
                                is_run_active_callback=is_run_active_callback,
                                heartbeat_callback=lock_heartbeat_callback
                            )
                            
                            log(f"[worker] [SNAPSHOT] Snapshot ingestion completed successfully")
                            
                        except Exception as ingest_error:
                            # Ingestion failed - error already logged and run marked as failed by ingest function
                            log(f"[worker] [SNAPSHOT] Snapshot ingestion failed: {str(ingest_error)}")
                            # The ingest function should have already marked the run as failed
                            # Just return here to stop processing
                            return
                        
                        # Check if run is still active after ingestion
                        if not is_run_active(supabase, run_id):
                            log(f"Run {run_id} is no longer active, stopping after ingestion")
                            return
                    else:
                        log(f"[worker] [SNAPSHOT] Snapshot ingestion disabled (SNAPSHOT_INGEST_ENABLED=false)")
                    
                    # Get list of all sheets (tabs) in the spreadsheet
                    try:
                        spreadsheet_metadata = service.spreadsheets().get(
                            spreadsheetId=sheet_id,
                            includeGridData=False
                        ).execute()
                        
                        sheets_list = spreadsheet_metadata.get('sheets', [])
                        tab_titles = [sheet['properties']['title'] for sheet in sheets_list]
                        log(f"[worker] [RECIPE] Found {len(tab_titles)} tabs in spreadsheet: {', '.join(tab_titles)}")
                    except Exception as e:
                        raise ValueError(f"[RECIPE] Failed to get spreadsheet metadata for sheet_id={sheet_id}: {str(e)}")
                    
                    # Load ALL tabs into work dictionary
                    work: Dict[str, List[Dict[str, Any]]] = {}
                    for tab_title in tab_titles:
                        rows = read_tab_as_rows(service, sheet_id, tab_title)
                        work[tab_title] = rows
                        log(f"[worker] [RECIPE] Loaded tab '{tab_title}': {len(rows)} rows")
                    
                    # Guard rail: Check if Master tab is missing or empty
                    master_rows = work.get("Master", [])
                    if not master_rows or len(master_rows) == 0:
                        raise ValueError(f"[RECIPE] Master tab is missing or empty in sheet {sheet_id}. Cannot run recipe without task definitions.")
                    
                    # Guard rail: Check if we have at least some data in one tab (excluding Master)
                    # This prevents silently running with empty work dict
                    non_master_rows = sum(len(rows) for tab_name, rows in work.items() if tab_name != "Master")
                    if non_master_rows == 0:
                        raise ValueError(f"[RECIPE] No data found in any tabs (excluding Master) in sheet {sheet_id}. Cannot run recipe with empty work dictionary.")
                    
                    # Check if run is still active before running recipe
                    if not is_run_active(supabase, run_id):
                        log(f"Run {run_id} is no longer active, stopping before running recipe")
                        return
                    
                    # Track last heartbeat time for recipe execution
                    last_recipe_heartbeat = time.time()
                    
                    # Define progress callback for real-time status updates
                    def recipe_progress_callback(row_index: int, status: str) -> None:
                        # 1) If run has been superseded, do nothing
                        if not is_run_active(supabase, run_id):
                            return
                        
                        # Heartbeat lock periodically during recipe execution (at least every 30-60 seconds)
                        nonlocal last_recipe_heartbeat
                        current_time = time.time()
                        if current_time - last_recipe_heartbeat >= 30.0:
                            try:
                                heartbeat_lock(supabase, project_id, run_id, WORKER_ID, RUN_LOCK_TTL_SECONDS)
                                last_recipe_heartbeat = current_time
                            except Exception as heartbeat_error:
                                log(f"Warning: Heartbeat failed during recipe execution: {str(heartbeat_error)}")
                                # Continue anyway - lock may still be valid
                        
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
                            work=work,
                            progress_callback=recipe_progress_callback
                        )
                    finally:
                        # Always restore original print
                        builtins.print = original_print
                    
                    # Check if run is still active after recipe execution
                    if not is_run_active(supabase, run_id):
                        log(f"Run {run_id} is no longer active, stopping before writing results")
                        return
                    
                    # Extract AI stats from result (always present, even if empty)
                    ai_run_stats = result.get("ai_run_stats", {
                        "total_cost_credits": 0.0,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0
                    })
                    ai_task_stats = result.get("ai_task_stats", [])
                    
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
                        update_data = {
                            "status": "failed",
                            "finished_at": finished_at.isoformat(),
                            "error_message": error_message,
                            "ai_total_cost_credits": ai_run_stats.get("total_cost_credits", 0.0),
                            "ai_total_prompt_tokens": ai_run_stats.get("total_prompt_tokens", 0),
                            "ai_total_completion_tokens": ai_run_stats.get("total_completion_tokens", 0)
                        }
                        supabase.table("runs").update(update_data).eq("id", run_id).eq("status", "running").execute()
                        
                        # Insert per-task stats (with error handling in case table doesn't exist)
                        _insert_task_stats(supabase, run_id, project_id, ai_task_stats)
                        
                        log(f"Failed recipe run {run_id} for project {project_id}: {error_message}")
                        return
                    
                    # Recipe succeeded - write results to Sheets
                    log(f"[worker] Recipe succeeded, writing results to Sheets")
                    
                    # Write back ALL sheets from work dictionary (except Master)
                    # This ensures any mutated sheets (e.g., DNC URL, DNC Email, Acct, VIP, etc.) are persisted
                    sheets_written = 0
                    for sheet_name, rows in work.items():
                        # Skip Master tab - it has special handling below
                        if sheet_name == "Master":
                            continue
                        
                        # Write the sheet back to Google Sheets
                        log(f"[worker] Writing {len(rows)} rows to '{sheet_name}' tab")
                        write_rows_to_tab(service, sheet_id, sheet_name, rows)
                        sheets_written += 1
                    
                    log(f"[worker] Wrote back {sheets_written} sheets to Google Sheets")
                    
                    # Update Master statuses (special handling - don't overwrite entire Master tab)
                    master_status_updates = result.get("master_status_updates", [])
                    
                    # Enrich master_status_updates with cost info for AI tasks
                    # Build a map of task_index -> cost_usd from ai_task_stats
                    task_cost_map = {}
                    ai_credits_to_usd = float(os.getenv("AI_CREDITS_TO_USD", "1.0"))
                    for task_stat in ai_task_stats:
                        task_index = task_stat.get("task_index")
                        cost_credits = task_stat.get("cost_credits", 0.0)
                        if task_index is not None:
                            task_cost_map[task_index] = cost_credits * ai_credits_to_usd
                    
                    # Add cost_usd to each update entry if it's an AI task
                    enriched_updates = []
                    for update in master_status_updates:
                        enriched_update = update.copy()
                        task_index = update.get("row_index")
                        if task_index in task_cost_map:
                            enriched_update["cost_usd"] = task_cost_map[task_index]
                        enriched_updates.append(enriched_update)
                    
                    if enriched_updates:
                        log(f"[worker] Updating {len(enriched_updates)} Master status rows (with cost info)")
                        update_master_statuses(service, sheet_id, "Master", enriched_updates)
                    
                    # Check if run is still active before marking as completed
                    if not is_run_active(supabase, run_id):
                        log(f"Run {run_id} is no longer active, stopping before marking completed")
                        return
                    
                    # Mark run as completed (store debug logs in error_message even on success)
                    finished_at = datetime.utcnow()
                    update_data = {
                        "status": "completed",
                        "finished_at": finished_at.isoformat(),
                        "ai_total_cost_credits": ai_run_stats.get("total_cost_credits", 0.0),
                        "ai_total_prompt_tokens": ai_run_stats.get("total_prompt_tokens", 0),
                        "ai_total_completion_tokens": ai_run_stats.get("total_completion_tokens", 0)
                    }
                    if debug_logs_text:
                        # Truncate if too long
                        update_data["error_message"] = debug_logs_text[:5000] if len(debug_logs_text) > 5000 else debug_logs_text
                    
                    supabase.table("runs").update(update_data).eq("id", run_id).eq("status", "running").execute()
                    
                    # Insert per-task stats (with error handling in case table doesn't exist)
                    _insert_task_stats(supabase, run_id, project_id, ai_task_stats)
                    
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
                    
                    # Try to extract AI stats from result if available (may not be if exception occurred before recipe completed)
                    ai_run_stats = {}
                    ai_task_stats = []
                    try:
                        if 'result' in locals():
                            ai_run_stats = result.get("ai_run_stats", {
                                "total_cost_credits": 0.0,
                                "total_prompt_tokens": 0,
                                "total_completion_tokens": 0
                            })
                            ai_task_stats = result.get("ai_task_stats", [])
                    except:
                        ai_run_stats = {
                            "total_cost_credits": 0.0,
                            "total_prompt_tokens": 0,
                            "total_completion_tokens": 0
                        }
                    
                    try:
                        update_data = {
                            "status": "failed",
                            "finished_at": finished_at.isoformat(),
                            "error_message": error_message,
                            "ai_total_cost_credits": ai_run_stats.get("total_cost_credits", 0.0),
                            "ai_total_prompt_tokens": ai_run_stats.get("total_prompt_tokens", 0),
                            "ai_total_completion_tokens": ai_run_stats.get("total_completion_tokens", 0)
                        }
                        supabase.table("runs").update(update_data).eq("id", run_id).eq("status", "running").execute()
                        
                        # Insert per-task stats (with error handling in case table doesn't exist)
                        _insert_task_stats(supabase, run_id, project_id, ai_task_stats)
                    except Exception as update_error:
                        log(f"Error updating run {run_id} to failed: {str(update_error)}")
                    
                    log(f"Failed recipe run {run_id} for project {project_id}: {error_message}")
            else:
                # Unsupported run_type - mark as failed with clear error message
                error_message = f"Unsupported run_type '{run_type}'. Enrichment workflow has been deprecated; use 'recipe' instead."
                log(f"Rejecting run {run_id} with unsupported run_type: {run_type}")
                
                finished_at = datetime.utcnow()
                try:
                    supabase.table("runs").update({
                        "status": "failed",
                        "finished_at": finished_at.isoformat(),
                        "error_message": error_message
                    }).eq("id", run_id).eq("status", "running").execute()
                except Exception as update_error:
                    log(f"Error updating run {run_id} to failed: {str(update_error)}")
                
                log(f"Marked run {run_id} as failed: {error_message}")
        finally:
            # Always release the lock when run ends (completed, failed, superseded, or early return)
            if lock_acquired:
                try:
                    release_lock(supabase, project_id, run_id, WORKER_ID)
                except Exception as release_error:
                    # Log but don't crash - release is best-effort cleanup
                    log(f"Warning: Failed to release lock for project {project_id}, run {run_id}: {str(release_error)}")


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


