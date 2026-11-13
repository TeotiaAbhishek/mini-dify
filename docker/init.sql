-- Runs once, when the Postgres data volume is first created.

-- The app's database (created by POSTGRES_DB) gets pgvector.
CREATE EXTENSION IF NOT EXISTS vector;

-- A separate database for the test suite, so tests never touch dev data.
CREATE DATABASE minidify_test;
\connect minidify_test
CREATE EXTENSION IF NOT EXISTS vector;
