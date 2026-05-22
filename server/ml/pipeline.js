/* ═══════════════════════════════════════════════════════
   FeD — ml/pipeline.js
   Full Face Recognition Pipeline
   Detect → Crop → Embed → Match against PostgreSQL
   ═══════════════════════════════════════════════════════ */
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const pool = require('../db/pool');
const {
  buildEmbeddingModel, extractEmbedding, cosineSimilarity,
  euclideanDistance, saveModel, loadModel, INPUT_SIZE, EMBED_DIM
} = require('./embedding_model');

const THRESHOLD = parseFloat(process.env.RECOGNITION_THRESHOLD || '0.62');

class FacePipeline {
  constructor() {
    this.embeddingModel = null;
    this.ready = false;
    this._embeddingCache = new Map(); // userId → [embedding]
  }

  /* ── Initialize models ─────────────────────────────── */
  async initialize() {
    console.log('[Pipeline] Initializing face recognition pipeline...');

    // Try loading saved model, else build fresh
    this.embeddingModel = await loadModel('embedding_model');
    if (!this.embeddingModel) {
      console.log('[Pipeline] No saved model found, building new embedding model...');
      this.embeddingModel = buildEmbeddingModel();
      console.log('[Pipeline] New model built:');
      this.embeddingModel.summary();
    }

    // Pre-load all embeddings from DB into cache
    await this._refreshCache();

    this.ready = true;
    console.log('[Pipeline] Pipeline ready.');
    return true;
  }

  /* ── Refresh embedding cache from DB ───────────────── */
  async _refreshCache() {
    try {
      const res = await pool.query(`
        SELECT fe.user_id, fe.embedding, u.name, u.roll_number, u.department
        FROM face_embeddings fe
        JOIN users u ON u.id = fe.user_id
        ORDER BY fe.user_id
      `);
      this._embeddingCache.clear();
      for (const row of res.rows) {
        const uid = row.user_id;
        if (!this._embeddingCache.has(uid)) {
          this._embeddingCache.set(uid, {
            userId: uid,
            name: row.name,
            roll: row.roll_number,
            dept: row.department,
            embeddings: []
          });
        }
        // PostgreSQL array comes as string '{0.1,0.2,...}', parse it
        let emb = row.embedding;
        if (typeof emb === 'string') {
          emb = emb.replace(/[{}]/g, '').split(',').map(Number);
        }
        this._embeddingCache.get(uid).embeddings.push(emb);
      }
      console.log(`[Pipeline] Cache loaded: ${this._embeddingCache.size} users, ${res.rows.length} embeddings`);
    } catch (err) {
      console.error('[Pipeline] Cache refresh error:', err.message);
    }
  }

  /* ── Register: extract embedding + store in DB ─────── */
  async registerFace(userId, imageBuffer) {
    if (!this.ready) throw new Error('Pipeline not initialized');

    // Preprocess image
    const processed = await sharp(imageBuffer)
      .resize(INPUT_SIZE, INPUT_SIZE, { fit: 'cover' })
      .removeAlpha()
      .raw()
      .toBuffer();

    // Extract embedding
    const embedding = await extractEmbedding(this.embeddingModel, imageBuffer);
    if (!embedding || embedding.length !== EMBED_DIM) {
      throw new Error('Failed to extract face embedding');
    }

    // Compute quality score (embedding magnitude, higher = more confident)
    const magnitude = Math.sqrt(embedding.reduce((s, v) => s + v * v, 0));

    // Store in PostgreSQL
    const pgArray = `{${embedding.join(',')}}`;
    await pool.query(
      `INSERT INTO face_embeddings (user_id, embedding, quality_score)
       VALUES ($1, $2, $3)`,
      [userId, pgArray, magnitude]
    );

    // Update cache
    await this._refreshCache();

    return { embedding, qualityScore: magnitude };
  }

  /* ── Recognize: match face against all stored embeddings */
  async recognizeFace(imageBuffer) {
    if (!this.ready) throw new Error('Pipeline not initialized');

    const embedding = await extractEmbedding(this.embeddingModel, imageBuffer);
    if (!embedding) return { matched: false, reason: 'no_embedding' };

    return this.matchEmbedding(embedding);
  }

  /* ── Match embedding against cache ─────────────────── */
  matchEmbedding(queryEmbedding) {
    let bestMatch  = null;
    let bestScore  = -1;
    let bestDist   = Infinity;

    for (const [userId, userData] of this._embeddingCache) {
      for (const storedEmb of userData.embeddings) {
        const sim  = cosineSimilarity(queryEmbedding, storedEmb);
        const dist = euclideanDistance(queryEmbedding, storedEmb);

        if (sim > bestScore) {
          bestScore = sim;
          bestDist  = dist;
          bestMatch = {
            userId:   userData.userId,
            name:     userData.name,
            roll:     userData.roll,
            dept:     userData.dept,
            similarity: sim,
            distance: dist
          };
        }
      }
    }

    if (!bestMatch || bestScore < THRESHOLD) {
      return {
        matched: false,
        bestScore,
        reason: bestMatch ? 'below_threshold' : 'no_users'
      };
    }

    return {
      matched: true,
      confidence: bestScore,
      distance: bestDist,
      user: bestMatch
    };
  }

  /* ── Retrain model with all registered faces ───────── */
  async retrainModel(options = {}) {
    const { trainEmbeddingModel } = require('./embedding_model');

    // Fetch all face images from DB
    const users = await pool.query(`
      SELECT u.id, u.photo FROM users u
      WHERE u.photo IS NOT NULL
    `);

    if (users.rows.length < 2) {
      return { success: false, reason: 'Need at least 2 users to train' };
    }

    // Prepare training data: convert photos to tensors
    const imagesByUser = {};
    for (const user of users.rows) {
      if (!user.photo) continue;

      // Decode base64 photo
      const base64 = user.photo.replace(/^data:image\/\w+;base64,/, '');
      const buf = Buffer.from(base64, 'base64');

      const processed = await sharp(buf)
        .resize(INPUT_SIZE, INPUT_SIZE, { fit: 'cover' })
        .removeAlpha()
        .raw()
        .toBuffer();

      const tensor = tf.tidy(() =>
        tf.tensor3d(new Uint8Array(processed), [INPUT_SIZE, INPUT_SIZE, 3]).div(255.0)
      );

      if (!imagesByUser[user.id]) imagesByUser[user.id] = [];
      imagesByUser[user.id].push(tensor);

      // Create augmented copies for more training data
      const { augmentFace } = require('./embedding_model');
      for (let i = 0; i < 4; i++) {
        const aug = augmentFace(tensor);
        imagesByUser[user.id].push(aug);
      }
    }

    const startTime = Date.now();
    const history = await trainEmbeddingModel(this.embeddingModel, imagesByUser, {
      epochs: options.epochs || 50,
      batchSize: options.batchSize || 16,
      margin: 0.3
    });

    const duration = Date.now() - startTime;

    // Save trained model
    await saveModel(this.embeddingModel, 'embedding_model');

    // Re-extract all embeddings with the updated model
    await this._recomputeAllEmbeddings();

    // Log training session
    const lastLoss = history.loss[history.loss.length - 1] || 0;
    await pool.query(
      `INSERT INTO training_sessions (model_version, loss, num_users, num_samples, epochs, duration_ms)
       VALUES ((SELECT COALESCE(MAX(model_version), 0) + 1 FROM training_sessions),
               $1, $2, $3, $4, $5)`,
      [lastLoss, users.rows.length, Object.values(imagesByUser).flat().length,
       options.epochs || 50, duration]
    );

    // Cleanup tensors
    for (const tensors of Object.values(imagesByUser)) {
      tensors.forEach(t => t.dispose());
    }

    return {
      success: true,
      epochs: options.epochs || 50,
      finalLoss: lastLoss,
      numUsers: users.rows.length,
      durationMs: duration
    };
  }

  /* ── Recompute all embeddings after retraining ─────── */
  async _recomputeAllEmbeddings() {
    const users = await pool.query(`SELECT id, photo FROM users WHERE photo IS NOT NULL`);

    for (const user of users.rows) {
      const base64 = user.photo.replace(/^data:image\/\w+;base64,/, '');
      const buf = Buffer.from(base64, 'base64');
      const embedding = await extractEmbedding(this.embeddingModel, buf);

      if (embedding) {
        const pgArray = `{${embedding.join(',')}}`;
        // Update existing or insert new
        await pool.query(`DELETE FROM face_embeddings WHERE user_id = $1`, [user.id]);
        await pool.query(
          `INSERT INTO face_embeddings (user_id, embedding, quality_score)
           VALUES ($1, $2, $3)`,
          [user.id, pgArray, Math.sqrt(embedding.reduce((s, v) => s + v * v, 0))]
        );
      }
    }

    await this._refreshCache();
  }

  /* ── Stats ─────────────────────────────────────────── */
  getStats() {
    return {
      ready: this.ready,
      cachedUsers: this._embeddingCache.size,
      totalEmbeddings: Array.from(this._embeddingCache.values())
        .reduce((sum, u) => sum + u.embeddings.length, 0),
      threshold: THRESHOLD,
      embeddingDim: EMBED_DIM
    };
  }
}

// Singleton instance
const pipeline = new FacePipeline();
module.exports = pipeline;
