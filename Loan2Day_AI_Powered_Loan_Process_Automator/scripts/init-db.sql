-- Loan2Day Database Initialization Script
-- This script sets up the initial database schema and indexes

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    income_in_cents DECIMAL(15,2) NOT NULL,
    employment_type VARCHAR(100) NOT NULL,
    credit_score INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Loan applications table
CREATE TABLE IF NOT EXISTS loan_applications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    amount_in_cents DECIMAL(15,2) NOT NULL,
    tenure_months INTEGER NOT NULL,
    purpose VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    fraud_score DECIMAL(5,4),
    trust_score DECIMAL(5,4),
    approved_amount_in_cents DECIMAL(15,2),
    approved_rate DECIMAL(5,4),
    approved_emi_in_cents DECIMAL(15,2),
    rejection_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Agent sessions table
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    current_step VARCHAR(50) NOT NULL,
    state_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- KYC documents table
CREATE TABLE IF NOT EXISTS kyc_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    document_type VARCHAR(100) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    ocr_text TEXT,
    sgs_score DECIMAL(5,4),
    verification_status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    extracted_data JSONB,
    trust_score DECIMAL(5,4),
    uploaded_at TIMESTAMP DEFAULT NOW(),
    verified_at TIMESTAMP
);

-- Audit log table for security events
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(100),
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Performance indexes for loan_applications table
CREATE INDEX IF NOT EXISTS idx_loan_applications_user_id ON loan_applications(user_id);
CREATE INDEX IF NOT EXISTS idx_loan_applications_status ON loan_applications(status);
CREATE INDEX IF NOT EXISTS idx_loan_applications_created_at ON loan_applications(created_at);
CREATE INDEX IF NOT EXISTS idx_loan_applications_amount ON loan_applications(amount_in_cents);

-- Performance indexes for agent_sessions table
CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_id ON agent_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_current_step ON agent_sessions(current_step);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_updated_at ON agent_sessions(updated_at);

-- Performance indexes for kyc_documents table
CREATE INDEX IF NOT EXISTS idx_kyc_documents_user_id ON kyc_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_kyc_documents_document_type ON kyc_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_kyc_documents_verification_status ON kyc_documents(verification_status);
CREATE INDEX IF NOT EXISTS idx_kyc_documents_uploaded_at ON kyc_documents(uploaded_at);

-- Performance indexes for audit_logs table
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_session_id ON audit_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_loan_applications_user_status ON loan_applications(user_id, status);
CREATE INDEX IF NOT EXISTS idx_kyc_documents_user_type ON kyc_documents(user_id, document_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_event ON audit_logs(user_id, event_type);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_loan_applications_updated_at BEFORE UPDATE ON loan_applications
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_sessions_updated_at BEFORE UPDATE ON agent_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for development (optional)
INSERT INTO users (name, phone, email, income_in_cents, employment_type, credit_score) VALUES
    ('Rajesh Kumar', '+91-9876543210', 'rajesh.kumar@example.com', 5000000, 'SALARIED', 750),
    ('Priya Sharma', '+91-9876543211', 'priya.sharma@example.com', 4500000, 'SALARIED', 720),
    ('Amit Patel', '+91-9876543212', 'amit.patel@example.com', 6000000, 'SELF_EMPLOYED', 680)
ON CONFLICT (phone) DO NOTHING;

-- Create database user for application (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'loan2day_app') THEN
        CREATE ROLE loan2day_app WITH LOGIN PASSWORD 'app_password';
    END IF;
END
$$;

-- Grant permissions to application user
GRANT CONNECT ON DATABASE loan2day TO loan2day_app;
GRANT USAGE ON SCHEMA public TO loan2day_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO loan2day_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO loan2day_app;

-- Grant permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO loan2day_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO loan2day_app;

-- Create indexes for JSONB columns (GIN indexes for better performance)
CREATE INDEX IF NOT EXISTS idx_agent_sessions_state_data_gin ON agent_sessions USING GIN (state_data);
CREATE INDEX IF NOT EXISTS idx_kyc_documents_extracted_data_gin ON kyc_documents USING GIN (extracted_data);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_data_gin ON audit_logs USING GIN (event_data);

-- Analyze tables for better query planning
ANALYZE users;
ANALYZE loan_applications;
ANALYZE agent_sessions;
ANALYZE kyc_documents;
ANALYZE audit_logs;

-- Log successful initialization
INSERT INTO audit_logs (event_type, event_data, created_at) VALUES
    ('DATABASE_INITIALIZED', '{"version": "1.0.0", "tables_created": 5, "indexes_created": 15}', NOW());

-- Display initialization summary
SELECT 
    'Database initialization completed successfully' as status,
    COUNT(*) as total_tables
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';