-- Migration: Add 'run_type' column to public.runs table
-- This migration adds support for different types of runs:
-- - 'enrichment' (default): existing enrichment workflow runs
-- - 'recipe': new recipe workflow runs (Master-sheet-driven)
-- 
-- To apply this migration:
-- 1. Connect to your Supabase database
-- 2. Run this SQL script in the SQL editor
-- 
-- The run_type column is NOT NULL with a default value of 'enrichment'
-- to ensure all existing rows are marked as enrichment runs.

ALTER TABLE public.runs
ADD COLUMN IF NOT EXISTS run_type text NOT NULL DEFAULT 'enrichment';

-- Add a comment to document the column
COMMENT ON COLUMN public.runs.run_type IS 'Type of run: enrichment (default) | recipe | (future types)';

