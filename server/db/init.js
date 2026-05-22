/* ═══════════════════════════════════════════════════════
   FeD — db/init.js
   Initialize database: create tables from schema.sql
   Run: npm run db:init
   ═══════════════════════════════════════════════════════ */
require('dotenv').config();
const fs   = require('fs');
const path = require('path');
const pool = require('./pool');

async function initDB() {
  console.log('[DB] Initializing database...');
  const schema = fs.readFileSync(path.join(__dirname, 'schema.sql'), 'utf8');
  try {
    await pool.query(schema);
    console.log('[DB] Schema applied successfully.');
    console.log('[DB] Tables: users, face_embeddings, attendance_records, training_sessions, model_weights');
  } catch (err) {
    console.error('[DB] Schema error:', err.message);
  } finally {
    await pool.end();
  }
}

initDB();
