-- Setup script for Supabase vector database for Odoo React Agent
-- This script creates the necessary database schema for storing and searching Odoo code embeddings

-- Enable the pgvector extension
create extension if not exists vector;

-- Create a table for storing Odoo code embeddings
create table odoo_embeddings (
  id bigint generated always as identity primary key,
  content text not null,
  embedding vector(768) not null,
  metadata jsonb not null,
  created_at timestamp with time zone default now()
);

-- Create an index for faster similarity search
create index on odoo_embeddings using ivfflat (embedding vector_l2_ops) with (lists = 100);

-- Create a function to match embeddings
create or replace function match_embeddings(
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    odoo_embeddings.id,
    odoo_embeddings.content,
    odoo_embeddings.metadata,
    1 - (odoo_embeddings.embedding <=> query_embedding) as similarity
  from odoo_embeddings
  where 1 - (odoo_embeddings.embedding <=> query_embedding) > match_threshold
  order by odoo_embeddings.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create indexes on common metadata fields for filtering
create index on odoo_embeddings ((metadata->>'module'));
create index on odoo_embeddings ((metadata->>'odoo_version'));
create index on odoo_embeddings ((metadata->>'file_type'));

-- Example query to test the function
-- select * from match_embeddings('[0.1, 0.2, ...]'::vector, 0.7, 5);
