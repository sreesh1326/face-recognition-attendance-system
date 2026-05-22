-- ═══════════════════════════════════════════════════════
-- FeD — PostgreSQL Schema
-- Supports 100+ users with face embeddings for 95%+ accuracy
-- ═══════════════════════════════════════════════════════

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(255) NOT NULL,
    roll_number   VARCHAR(100) UNIQUE NOT NULL,
    department    VARCHAR(255) DEFAULT 'N/A',
    photo         TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Face embeddings — multiple 128-d vectors per user for accuracy
CREATE TABLE IF NOT EXISTS face_embeddings (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding     DOUBLE PRECISION[128] NOT NULL,
    quality_score DOUBLE PRECISION DEFAULT 0.0,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Attendance records
CREATE TABLE IF NOT EXISTS attendance_records (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    roll_number   VARCHAR(100) NOT NULL,
    department    VARCHAR(255),
    confidence    DOUBLE PRECISION NOT NULL,
    marked_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Training session logs
CREATE TABLE IF NOT EXISTS training_sessions (
    id            SERIAL PRIMARY KEY,
    model_version INTEGER NOT NULL DEFAULT 1,
    loss          DOUBLE PRECISION,
    accuracy      DOUBLE PRECISION,
    num_users     INTEGER,
    num_samples   INTEGER,
    epochs        INTEGER,
    duration_ms   INTEGER,
    trained_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Model weights storage
CREATE TABLE IF NOT EXISTS model_weights (
    id            SERIAL PRIMARY KEY,
    model_name    VARCHAR(100) NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    weights_json  JSONB NOT NULL,
    topology_json JSONB NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_name, version)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_embed_user   ON face_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_att_user     ON attendance_records(user_id);
CREATE INDEX IF NOT EXISTS idx_att_time     ON attendance_records(marked_at);
CREATE INDEX IF NOT EXISTS idx_users_roll   ON users(roll_number);
CREATE INDEX IF NOT EXISTS idx_att_roll     ON attendance_records(roll_number);
