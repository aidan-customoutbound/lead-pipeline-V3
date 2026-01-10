"""
FastAPI web server for receiving HTTP POST requests from Google Sheets.

Endpoints:
- POST /upload: Trigger upload_csv.py for a given project_id
- POST /start: Trigger enrich_workflow.py for a given project_id
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

# Import the run functions from our modules
import upload_csv
import enrich_workflow

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


@app.post("/start")
async def start_endpoint(request: Request) -> JSONResponse:
    """
    Start enrichment workflow for a given project_id.
    
    Request body (JSON):
    {
        "project_id": "...",
        "secret": "..."
    }
    
    Returns:
        JSON response with status and project_id
    """
    try:
        # Parse request body
        body = await request.json()
        project_id = body.get("project_id")
        secret = body.get("secret")
        
        # Log request
        timestamp = datetime.utcnow().isoformat()
        logger.info(f"[{timestamp}] POST /start - project_id={project_id}")
        
        # Validate required fields
        if not project_id:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: project_id"
            )
        
        # Validate secret
        validate_secret(secret)
        
        # Call enrich_workflow.run() inline (not background)
        await enrich_workflow.run(project_id)
        
        logger.info(f"[{timestamp}] POST /start - SUCCESS - project_id={project_id}")
        
        return JSONResponse(content={
            "status": "started",
            "project_id": project_id
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

