-- Migration: Add 'sheet_id' column to prompts table
-- This migration adds support for per-project Google Sheet IDs
-- 
-- To apply this migration:
-- 1. Connect to your Supabase database
-- 2. Run this SQL script in the SQL editor
-- 
-- The sheet_id column is optional (nullable) to maintain backward compatibility
-- with existing prompts that don't have a sheet_id set.

ALTER TABLE public.prompts
ADD COLUMN IF NOT EXISTS sheet_id TEXT;

-- Add a comment to document the column
COMMENT ON COLUMN public.prompts.sheet_id IS 'Google Sheet ID for exporting results. Extracted from sheet_url during CSV upload.';

