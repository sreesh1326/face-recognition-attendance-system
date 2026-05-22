/* ═══════════════════════════════════════════════════════
   FeD — server.js
   Express REST API for face recognition + PostgreSQL
   ═══════════════════════════════════════════════════════ */
require('dotenv').config();
const express  = require('express');
const cors     = require('cors');
const path     = require('path');
const pool     = require('./db/pool');
const pipeline = require('./ml/pipeline');

const app  = express();
const PORT = process.env.PORT || 3000;

/* ── Middleware ───────────────────────────────────────── */
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve frontend static files
app.use(express.static(path.join(__dirname, '..')));

/* ═══════════ USER ROUTES ═══════════════════════════════ */

// GET all users
app.get('/api/users', async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT id, name, roll_number, department, created_at
       FROM users ORDER BY created_at DESC`
    );
    res.json({ success: true, users: result.rows });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET single user
app.get('/api/users/:roll', async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT id, name, roll_number, department, created_at
       FROM users WHERE roll_number = $1`, [req.params.roll]
    );
    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    res.json({ success: true, user: result.rows[0] });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST register new user + face
app.post('/api/users/register', async (req, res) => {
  const { name, roll, dept, photo } = req.body;

  if (!name || !roll) {
    return res.status(400).json({ success: false, error: 'Name and roll are required' });
  }
  if (!photo) {
    return res.status(400).json({ success: false, error: 'Face photo is required' });
  }

  try {
    // Check duplicate
    const existing = await pool.query(
      `SELECT id FROM users WHERE roll_number = $1`, [roll]
    );
    if (existing.rows.length > 0) {
      return res.status(409).json({ success: false, error: `Roll "${roll}" already registered` });
    }

    // Insert user
    const userRes = await pool.query(
      `INSERT INTO users (name, roll_number, department, photo)
       VALUES ($1, $2, $3, $4) RETURNING id, name, roll_number, department, created_at`,
      [name, roll, dept || 'N/A', photo]
    );
    const user = userRes.rows[0];

    // Extract + store face embedding
    const base64 = photo.replace(/^data:image\/\w+;base64,/, '');
    const imgBuf = Buffer.from(base64, 'base64');
    const embResult = await pipeline.registerFace(user.id, imgBuf);

    res.json({
      success: true,
      user,
      embedding: {
        qualityScore: embResult.qualityScore,
        dimensions: embResult.embedding.length
      }
    });
  } catch (err) {
    console.error('[API] Register error:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// DELETE user
app.delete('/api/users/:roll', async (req, res) => {
  try {
    const result = await pool.query(
      `DELETE FROM users WHERE roll_number = $1 RETURNING name`, [req.params.roll]
    );
    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    res.json({ success: true, deleted: result.rows[0].name });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

/* ═══════════ RECOGNITION ROUTES ════════════════════════ */

// POST recognize face from image
app.post('/api/recognize', async (req, res) => {
  const { photo } = req.body;
  if (!photo) {
    return res.status(400).json({ success: false, error: 'Photo required' });
  }

  try {
    const base64 = photo.replace(/^data:image\/\w+;base64,/, '');
    const imgBuf = Buffer.from(base64, 'base64');
    const result = await pipeline.recognizeFace(imgBuf);
    res.json({ success: true, ...result });
  } catch (err) {
    console.error('[API] Recognize error:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

/* ═══════════ ATTENDANCE ROUTES ═════════════════════════ */

// POST mark attendance
app.post('/api/attendance', async (req, res) => {
  const { userId, name, roll, dept, confidence } = req.body;
  if (!userId || !roll) {
    return res.status(400).json({ success: false, error: 'userId and roll required' });
  }

  try {
    // Check cooldown (no duplicate within 60s)
    const recent = await pool.query(
      `SELECT id FROM attendance_records
       WHERE user_id = $1 AND marked_at > NOW() - INTERVAL '60 seconds'`,
      [userId]
    );
    if (recent.rows.length > 0) {
      return res.json({ success: true, duplicate: true, message: 'Already marked recently' });
    }

    const result = await pool.query(
      `INSERT INTO attendance_records (user_id, name, roll_number, department, confidence)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING id, marked_at`,
      [userId, name, roll, dept || 'N/A', confidence || 0]
    );

    res.json({ success: true, record: result.rows[0] });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET attendance records
app.get('/api/attendance', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 200;
    const result = await pool.query(
      `SELECT id, name, roll_number, department, confidence, marked_at
       FROM attendance_records ORDER BY marked_at DESC LIMIT $1`, [limit]
    );
    res.json({ success: true, records: result.rows });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET today's attendance
app.get('/api/attendance/today', async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT id, name, roll_number, department, confidence, marked_at
       FROM attendance_records
       WHERE marked_at::date = CURRENT_DATE
       ORDER BY marked_at DESC`
    );
    res.json({ success: true, records: result.rows });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// DELETE clear all attendance
app.delete('/api/attendance', async (req, res) => {
  try {
    await pool.query(`DELETE FROM attendance_records`);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

/* ═══════════ TRAINING ROUTES ═══════════════════════════ */

// POST trigger model retraining
app.post('/api/train', async (req, res) => {
  const { epochs, batchSize } = req.body;
  try {
    const result = await pipeline.retrainModel({
      epochs: epochs || 50,
      batchSize: batchSize || 16
    });
    res.json({ success: true, ...result });
  } catch (err) {
    console.error('[API] Train error:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// GET training history
app.get('/api/train/history', async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT * FROM training_sessions ORDER BY trained_at DESC LIMIT 20`
    );
    res.json({ success: true, sessions: result.rows });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

/* ═══════════ STATS ROUTE ═══════════════════════════════ */

app.get('/api/stats', async (req, res) => {
  try {
    const users = await pool.query(`SELECT COUNT(*) as count FROM users`);
    const att   = await pool.query(`SELECT COUNT(*) as count FROM attendance_records`);
    const today = await pool.query(
      `SELECT COUNT(*) as count FROM attendance_records
       WHERE marked_at::date = CURRENT_DATE`
    );
    const pipeStats = pipeline.getStats();

    res.json({
      success: true,
      totalUsers: parseInt(users.rows[0].count),
      totalAttendance: parseInt(att.rows[0].count),
      todayAttendance: parseInt(today.rows[0].count),
      pipeline: pipeStats
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

/* ═══════════ START SERVER ══════════════════════════════ */

async function start() {
  try {
    // Test DB connection
    await pool.query('SELECT NOW()');
    console.log('[DB] PostgreSQL connected');

    // Initialize ML pipeline
    await pipeline.initialize();

    app.listen(PORT, () => {
      console.log(`\n══════════════════════════════════════`);
      console.log(`  FeD Server running on port ${PORT}`);
      console.log(`  http://localhost:${PORT}`);
      console.log(`══════════════════════════════════════\n`);
    });
  } catch (err) {
    console.error('[Server] Startup failed:', err.message);
    console.error('Make sure PostgreSQL is running and the database exists.');
    console.error('Run: createdb fed_attendance && npm run db:init');
    process.exit(1);
  }
}

start();
