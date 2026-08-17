-- =============================================================================
-- PostgreSQL initialization script
-- =============================================================================
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "hstore";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS legalai;
SET search_path = legalai, public;

-- Roles
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'legalai_app') THEN
        CREATE ROLE legalai_app WITH LOGIN PASSWORD 'legalai_app_pass';
    END IF;
END
$$;

-- The main application user gets full access
GRANT ALL ON SCHEMA legalai TO legalai;
GRANT ALL ON SCHEMA public TO legalai;
