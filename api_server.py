"""
FastAPI web server for receiving HTTP POST requests from Google Sheets.

Endpoints:
- POST /upload: Trigger upload_csv.py for a given project_id
- POST /start: Queue enrichment workflow for a given project_id
- POST /stop: Cancel any queued or running runs for a given project_id
"""

import csv
import io
import logging
import os
from datetime import datetime
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from supabase import create_client, Client

# Import the run functions from our modules
import upload_csv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Lead Pipeline API Server")

# Get secret from environment
GOOGLE_PUSH_SECRET = os.getenv("GOOGLE_PUSH_SECRET")

# Supabase configuration for runs table
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def get_supabase_client() -> Client:
    """
    Create and return a Supabase client for the runs table.
    
    Returns:
        Supabase client instance
        
    Raises:
        HTTPException: If Supabase configuration is missing
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL or SUPABASE_KEY environment variables are not set")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: Supabase credentials not configured"
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def validate_secret(secret: str) -> None:
    """
    Validate the provided secret against the environment variable.
    
    Args:
        secret: The secret to validate
        
    Raises:
        HTTPException: If secret is missing or doesn't match
    """
    if not GOOGLE_PUSH_SECRET:
        logger.error("GOOGLE_PUSH_SECRET environment variable is not set")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GOOGLE_PUSH_SECRET not configured"
        )
    
    if not secret:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: secret"
        )
    
    if secret != GOOGLE_PUSH_SECRET:
        logger.warning(f"Invalid secret provided (timestamp: {datetime.utcnow().isoformat()})")
        raise HTTPException(
            status_code=403,
            detail="Invalid secret"
        )


def fetch_csv_from_url(sheet_url: str) -> List[Dict[str, str]]:
    """
    Fetch CSV data from a Google Sheets export URL.
    
    Args:
        sheet_url: Google Sheets CSV export URL
        
    Returns:
        List of dictionaries representing CSV rows
        
    Raises:
        HTTPException: If fetch fails or CSV is invalid
    """
    try:
        logger.info(f"Fetching CSV from URL: {sheet_url}")
        response = requests.get(sheet_url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV content
        csv_content = response.text
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        
        # Convert to list of dicts
        csv_rows = list(reader)
        
        if not csv_rows:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty or has no data rows"
            )
        
        logger.info(f"Successfully fetched {len(csv_rows)} rows from CSV")
        return csv_rows
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching CSV from URL: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch CSV from URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV: {str(e)}"
        )


@app.post("/upload")
async def upload_endpoint(request: Request) -> JSONResponse:
    """
    Upload CSV data from Google Sheets to Supabase.
    
    Request body (JSON):
    {
        "project_id": "...",
        "sheet_url": "...",
        "secret": "..."
    }
    
    Returns:
        JSON response with status and upload results
    """
    try:
        # Parse request body
        body = await request.json()
        project_id = body.get("project_id")
        sheet_url = body.get("sheet_url")
        secret = body.get("secret")
        
        # Log request
        timestamp = datetime.utcnow().isoformat()
        logger.info(f"[{timestamp}] POST /upload - project_id={project_id}")
        
        # Validate required fields
        if not project_id:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: project_id"
            )
        
        if not sheet_url:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: sheet_url"
            )
        
        # Validate secret
        validate_secret(secret)
        
        # Fetch CSV from URL
        csv_rows = fetch_csv_from_url(sheet_url)
        
        # Call upload_csv.run() with sheet_url to extract and store sheet_id
        result = upload_csv.run(project_id, csv_rows, sheet_url=sheet_url)
        
        logger.info(f"[{timestamp}] POST /upload - SUCCESS - project_id={project_id} rows={result.get('rows', 0)}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in /upload endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Contract:
# - Starting a new run for a project_id will mark any existing runs for that project
#   with status in ('queued', 'running') as 'superseded'.
# - The worker treats 'superseded' as a signal to abort the run as soon as possible
#   to avoid unnecessary Exa/LLM spend.
@app.post("/start")
async def start_endpoint(request: Request) -> JSONResponse:
    """
    Queue enrichment workflow for a given project_id.
    
    Request body (JSON):
    {
        "project_id": "...",
        "secret": "...",
        "run_type": "..." (optional, defaults to "enrichment")
    }
    
    Returns:
        JSON response with status='queued', project_id, and run_id
    """
    try:
        # Parse request body
        body = await request.json()
        project_id = body.get("project_id")
        secret = body.get("secret")
        run_type = body.get("run_type", "enrichment")
        
        # Default to 'enrichment' if run_type is not provided or blank
        if not run_type or not run_type.strip():
            run_type = "enrichment"
        else:
            run_type = run_type.strip()
        
        # Log request
        timestamp = datetime.utcnow().isoformat()
        logger.info(f"[{timestamp}] POST /start - project_id={project_id}, run_type={run_type}")
        
        # Validate required fields
        if not project_id or not project_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Missing or empty required field: project_id"
            )
        
        # Validate secret
        validate_secret(secret)
        
        # Create Supabase client for runs table
        supabase = get_supabase_client()
        
        # Supersede existing runs for this project_id that are queued or running
        finished_at = datetime.utcnow()
        try:
            supersede_response = supabase.table("runs").update({
                "status": "superseded",
                "finished_at": finished_at.isoformat()
            }).eq("project_id", project_id).in_("status", ["queued", "running"]).execute()
            
            superseded_count = len(supersede_response.data) if supersede_response.data else 0
            if superseded_count > 0:
                logger.info(f"Superseded {superseded_count} existing queued/running run(s) for project_id={project_id}")
            else:
                logger.info(f"No existing queued/running runs to supersede for project_id={project_id}")
        except Exception as e:
            logger.warning(f"Error superseding existing runs: {str(e)}", exc_info=True)
            # Continue anyway - this is not critical
        
        # Insert a new run row with status='queued'
        try:
            run_insert_response = supabase.table("runs").insert({
                "project_id": project_id,
                "status": "queued",
                "run_type": run_type,
                "started_at": None,
                "finished_at": None,
                "run_token": None,
                "total_prospects": None,
                "prospects_enriched": None,
                "error_message": None
            }).execute()
            
            if not run_insert_response.data or len(run_insert_response.data) == 0:
                logger.error("Failed to insert run row: no data returned")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create run record"
                )
            
            run_id = run_insert_response.data[0]["id"]
            logger.info(f"Created queued run record: run_id={run_id}, project_id={project_id}")
            
        except Exception as e:
            logger.error(f"Error inserting run row: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create run record: {str(e)}"
            )
        
        logger.info(f"[{timestamp}] POST /start - SUCCESS - project_id={project_id}, run_id={run_id}")
        
        return JSONResponse(content={
            "project_id": project_id,
            "run_id": run_id,
            "status": "queued"
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in /start endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/stop")
async def stop_endpoint(request: Request) -> JSONResponse:
    """
    Cancel any queued or running runs for a given project_id.
    
    Request body (JSON):
    {
        "project_id": "...",
        "secret": "..."
    }
    
    Returns:
        JSON response with project_id and stopped_runs count
    """
    try:
        # Parse request body
        body = await request.json()
        project_id = body.get("project_id")
        secret = body.get("secret")
        
        # Log request
        timestamp = datetime.utcnow().isoformat()
        logger.info(f"[{timestamp}] POST /stop - project_id={project_id}")
        
        # Validate required fields
        if not project_id or not project_id.strip():
            raise HTTPException(
                status_code=400,
                detail="project_id is required"
            )
        
        # Validate secret
        validate_secret(secret)
        
        # Create Supabase client for runs table
        supabase = get_supabase_client()
        
        # Update runs table: mark queued/running runs as superseded
        finished_at = datetime.utcnow()
        try:
            stop_response = supabase.table("runs").update({
                "status": "superseded",
                "finished_at": finished_at.isoformat(),
                "error_message": "Stopped via /stop endpoint"
            }).eq("project_id", project_id).in_("status", ["queued", "running"]).execute()
            
            stopped_count = len(stop_response.data) if stop_response.data else 0
            logger.info(f"Stopped {stopped_count} queued/running run(s) for project_id={project_id}")
            
        except Exception as e:
            logger.error(f"Error stopping runs: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to stop runs"
            )
        
        logger.info(f"[{timestamp}] POST /stop - SUCCESS - project_id={project_id}, stopped_runs={stopped_count}")
        
        return JSONResponse(content={
            "project_id": project_id,
            "stopped_runs": stopped_count
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in /stop endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint.
    
    Returns:
        JSON response with status
    """
    return JSONResponse(content={"status": "ok"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

