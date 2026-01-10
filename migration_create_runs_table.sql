create table if not exists public.runs (
  id                  bigserial primary key,
  project_id          text        not null,
  status              text        not null default 'queued',
  run_token           text,
  created_at          timestamptz not null default now(),
  started_at          timestamptz,
  finished_at         timestamptz,
  total_prospects     integer,
  prospects_enriched  integer,
  error_message       text
);

create index if not exists runs_project_status_idx
  on public.runs (project_id, status);

create index if not exists runs_created_at_idx
  on public.runs (created_at desc);

comment on column public.runs.status is
  'Run status: queued | running | completed | failed | superseded';

