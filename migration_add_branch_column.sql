-- Migration: Add 'branch' column to prompts table
-- This migration adds support for Branch (IF/THEN) prompt selection functionality
-- 
-- To apply this migration:
-- 1. Connect to your Supabase database
-- 2. Run this SQL script in the SQL editor
-- 
-- The branch column is optional (nullable) to maintain backward compatibility
-- with existing prompts that don't use Branch functionality.

ALTER TABLE public.prompts
ADD COLUMN IF NOT EXISTS branch TEXT;

-- Add a comment to document the column
COMMENT ON COLUMN public.prompts.branch IS 'Optional Branch DSL for conditional prompt selection. Format: BRANCH({placeholder}): "match1" :: prompt1 "match2" :: prompt2 ELSE :: prompt_else';

