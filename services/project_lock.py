"""
Project lock module for coordinating run execution across multiple workers.

This module provides functions to acquire, heartbeat, and release locks on projects
to ensure only one run per project executes at a time.
"""

import os
from typing import Optional
from supabase import Client


def log(message: str) -> None:
    """Print a log message with [LOCK] prefix."""
    print(f"[LOCK] {message}")


def acquire_lock(
    supabase: Client,
    project_id: str,
    run_id: int,
    worker_id: str,
    ttl_seconds: int
) -> bool:
    """
    Acquire a lock for a project.
    
    This function calls the Supabase RPC function acquire_project_lock to atomically
    acquire a lock for the given project. Only one run per project can hold the lock
    at a time.
    
    Args:
        supabase: Supabase client instance
        project_id: Project ID to lock
        run_id: Run ID attempting to acquire the lock
        worker_id: Worker identifier (e.g., "hostname:pid")
        ttl_seconds: Time-to-live for the lock in seconds
        
    Returns:
        True if lock was acquired, False if lock is already held by another run
        
    Raises:
        Exception: If RPC call fails (transient Supabase issue)
    """
    try:
        response = supabase.rpc(
            "acquire_project_lock",
            {
                "p_project_id": project_id,
                "p_run_id": run_id,
                "p_ttl_seconds": ttl_seconds,
                "p_worker_id": worker_id
            }
        ).execute()
        
        # Parse boolean return value
        # Supabase RPC returns data as a list, and the boolean is the first element
        if response.data is None:
            raise Exception("RPC call returned None")
        
        # Handle different possible response formats
        if isinstance(response.data, bool):
            result = response.data
        elif isinstance(response.data, list) and len(response.data) > 0:
            result = response.data[0]
        elif isinstance(response.data, dict):
            # Some RPC functions return a dict with the result
            result = response.data.get("result", response.data.get("acquire_project_lock", False))
        else:
            # Try to convert to bool
            result = bool(response.data)
        
        if result:
            log(f"Lock acquired for project {project_id}, run {run_id}, worker {worker_id}")
        else:
            log(f"Lock busy for project {project_id}, run {run_id} (held by another run)")
        
        return bool(result)
        
    except Exception as e:
        error_msg = f"Failed to acquire lock for project {project_id}, run {run_id}: {str(e)}"
        log(f"ERROR: {error_msg}")
        raise Exception(error_msg)


def heartbeat_lock(
    supabase: Client,
    project_id: str,
    run_id: int,
    worker_id: str,
    ttl_seconds: int
) -> bool:
    """
    Refresh the heartbeat for a project lock.
    
    This function calls the Supabase RPC function heartbeat_project_lock to extend
    the lock's TTL. Should be called periodically during long-running operations.
    
    Args:
        supabase: Supabase client instance
        project_id: Project ID for the lock
        run_id: Run ID holding the lock
        worker_id: Worker identifier
        ttl_seconds: Time-to-live to extend the lock to
        
    Returns:
        True if heartbeat succeeded, False if lock is no longer held by this run
        
    Raises:
        Exception: If RPC call fails (transient Supabase issue)
    """
    try:
        response = supabase.rpc(
            "heartbeat_project_lock",
            {
                "p_project_id": project_id,
                "p_run_id": run_id,
                "p_ttl_seconds": ttl_seconds,
                "p_worker_id": worker_id
            }
        ).execute()
        
        # Parse boolean return value
        if response.data is None:
            raise Exception("RPC call returned None")
        
        # Handle different possible response formats
        if isinstance(response.data, bool):
            result = response.data
        elif isinstance(response.data, list) and len(response.data) > 0:
            result = response.data[0]
        elif isinstance(response.data, dict):
            result = response.data.get("result", response.data.get("heartbeat_project_lock", False))
        else:
            result = bool(response.data)
        
        if result:
            log(f"Heartbeat OK for project {project_id}, run {run_id}, worker {worker_id}")
        else:
            log(f"Heartbeat failed for project {project_id}, run {run_id} (lock no longer held)")
        
        return bool(result)
        
    except Exception as e:
        error_msg = f"Failed to heartbeat lock for project {project_id}, run {run_id}: {str(e)}"
        log(f"ERROR: {error_msg}")
        raise Exception(error_msg)


def release_lock(
    supabase: Client,
    project_id: str,
    run_id: int,
    worker_id: str
) -> bool:
    """
    Release a project lock.
    
    This function calls the Supabase RPC function release_project_lock to release
    the lock held by this run. Should be called in a finally block to ensure cleanup.
    
    Args:
        supabase: Supabase client instance
        project_id: Project ID for the lock
        run_id: Run ID holding the lock
        worker_id: Worker identifier
        
    Returns:
        True if release succeeded, False otherwise
        
    Raises:
        Exception: If RPC call fails (transient Supabase issue)
    """
    try:
        response = supabase.rpc(
            "release_project_lock",
            {
                "p_project_id": project_id,
                "p_run_id": run_id,
                "p_worker_id": worker_id
            }
        ).execute()
        
        # Parse boolean return value
        if response.data is None:
            raise Exception("RPC call returned None")
        
        # Handle different possible response formats
        if isinstance(response.data, bool):
            result = response.data
        elif isinstance(response.data, list) and len(response.data) > 0:
            result = response.data[0]
        elif isinstance(response.data, dict):
            result = response.data.get("result", response.data.get("release_project_lock", False))
        else:
            result = bool(response.data)
        
        if result:
            log(f"Lock released for project {project_id}, run {run_id}, worker {worker_id}")
        else:
            log(f"Lock release failed for project {project_id}, run {run_id} (may not have been held)")
        
        return bool(result)
        
    except Exception as e:
        error_msg = f"Failed to release lock for project {project_id}, run {run_id}: {str(e)}"
        log(f"ERROR: {error_msg}")
        # Don't raise - release is best-effort cleanup
        return False
