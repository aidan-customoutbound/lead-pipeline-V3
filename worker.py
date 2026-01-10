"""
Worker process that polls for queued runs and processes them.

This worker continuously polls the public.runs table for runs with status='queued',
claims them atomically, and executes the enrichment workflow.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from api_server import get_supabase_client

import enrich_workflow

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


def claim_next_run(supabase: Client) -> Optional[Dict[str, Any]]:
    """
    Atomically claim the oldest queued run.
    
    This function:
    1. Finds the oldest run with status='queued'
    2. Atomically updates it to status='running' and sets started_at
    3. Returns the full row (id + project_id) if successful
    4. Returns None if no queued runs are found
    
    Args:
        supabase: Supabase client instance
        
    Returns:
        Dictionary with run data (id, project_id) if a run was claimed, None otherwise
    """
    try:
        # Find the oldest queued run
        response = supabase.table("runs").select("id, project_id").eq("status", "queued").order("created_at", desc=False).limit(1).execute()
        
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
        
        log(f"Claimed run {run_id} for project {run['project_id']}")
        return run
        
    except Exception as e:
        log(f"Error claiming run: {str(e)}")
        return None


async def process_run(run_row, supabase):
    """
    Process a claimed run by executing the enrichment workflow.
    
    Args:
        supabase: Supabase client instance
        run: Dictionary with run data (id, project_id)
    """
    run_id = run_row["id"]
    project_id = run_row["project_id"]
    
    log(f"processing run {run_id} for project {project_id}")
    
    try:
        # Run the enrichment workflow
        await enrich_workflow.run(project_id)
        
        # Update run to completed
        finished_at = datetime.utcnow()
        supabase.table("runs").update({
            "status": "completed",
            "finished_at": finished_at.isoformat()
        }).eq("id", run_id).execute()
        
        log(f"Completed run {run_id} for project {project_id}")
        
    except Exception as e:
        # Update run to failed
        finished_at = datetime.utcnow()
        error_message = str(e)[:500]  # Truncate to first 500 chars
        
        try:
            supabase.table("runs").update({
                "status": "failed",
                "finished_at": finished_at.isoformat(),
                "error_message": error_message
            }).eq("id", run_id).execute()
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
            await process_run(supabase, run)
            
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


