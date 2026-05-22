/* ═══════════════════════════════════════════════════════
   FeD — db/pool.js
   PostgreSQL connection pool + query helpers
   ═══════════════════════════════════════════════════════ */
const { Pool } = require('pg');

const pool = new Pool({
  host:     process.env.PGHOST     || 'localhost',
  port:     parseInt(process.env.PGPORT || '5432'),
  user:     process.env.PGUSER     || 'postgres',
  password: process.env.PGPASSWORD || 'postgres',
  database: process.env.PGDATABASE || 'fed_attendance',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000
});

pool.on('error', (err) => {
  console.error('[DB] Unexpected pool error:', err);
});

module.exports = pool;
