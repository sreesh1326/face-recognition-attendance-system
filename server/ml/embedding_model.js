/* ═══════════════════════════════════════════════════════
   FeD — ml/embedding_model.js
   Custom Face Embedding CNN — 128-dim face descriptor
   Architecture: ResNet-inspired with residual blocks
   Training: Triplet loss for discriminative embeddings
   Target: 95%+ recognition accuracy for 100+ users
   ═══════════════════════════════════════════════════════ */
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs   = require('fs');

const INPUT_SIZE   = parseInt(process.env.FACE_INPUT_SIZE || '160');
const EMBED_DIM    = parseInt(process.env.EMBEDDING_DIM   || '128');
const SAVE_DIR     = process.env.MODEL_SAVE_PATH || './ml/saved_models';

/* ── Residual Block ──────────────────────────────────── */
function residualBlock(input, filters, stride = 1) {
  let x = tf.layers.conv2d({
    filters, kernelSize: 3, strides: stride,
    padding: 'same', useBias: false
  }).apply(input);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.activation({ activation: 'relu' }).apply(x);

  x = tf.layers.conv2d({
    filters, kernelSize: 3, strides: 1,
    padding: 'same', useBias: false
  }).apply(x);
  x = tf.layers.batchNormalization().apply(x);

  // Shortcut connection
  let shortcut = input;
  if (stride !== 1 || input.shape[input.shape.length - 1] !== filters) {
    shortcut = tf.layers.conv2d({
      filters, kernelSize: 1, strides: stride,
      padding: 'same', useBias: false
    }).apply(input);
    shortcut = tf.layers.batchNormalization().apply(shortcut);
  }

  x = tf.layers.add().apply([x, shortcut]);
  x = tf.layers.activation({ activation: 'relu' }).apply(x);
  return x;
}

/* ── Build Embedding Model ───────────────────────────── */
function buildEmbeddingModel() {
  const input = tf.input({ shape: [INPUT_SIZE, INPUT_SIZE, 3], name: 'face_input' });

  // Stem: initial convolution
  let x = tf.layers.conv2d({
    filters: 64, kernelSize: 7, strides: 2,
    padding: 'same', useBias: false, name: 'stem_conv'
  }).apply(input);
  x = tf.layers.batchNormalization({ name: 'stem_bn' }).apply(x);
  x = tf.layers.activation({ activation: 'relu' }).apply(x);
  x = tf.layers.maxPooling2d({ poolSize: 3, strides: 2, padding: 'same' }).apply(x);

  // Residual stages
  x = residualBlock(x, 64,  1);
  x = residualBlock(x, 64,  1);
  x = residualBlock(x, 128, 2);
  x = residualBlock(x, 128, 1);
  x = residualBlock(x, 256, 2);
  x = residualBlock(x, 256, 1);
  x = residualBlock(x, 512, 2);
  x = residualBlock(x, 512, 1);

  // Head: global pool → dense → L2 normalize
  x = tf.layers.globalAveragePooling2d().apply(x);
  x = tf.layers.dropout({ rate: 0.4 }).apply(x);
  x = tf.layers.dense({ units: 256, activation: 'relu', name: 'fc1' }).apply(x);
  x = tf.layers.dropout({ rate: 0.3 }).apply(x);

  // Raw embedding
  let embed = tf.layers.dense({ units: EMBED_DIM, name: 'embedding_raw' }).apply(x);

  // L2 Normalization layer (custom lambda)
  const output = tf.layers.lambda({
    func: (t) => tf.div(t, tf.add(tf.norm(t, 2, -1, true), 1e-10)),
    name: 'l2_normalize'
  }).apply(embed);

  const model = tf.model({ inputs: input, outputs: output, name: 'FaceEmbeddingNet' });
  return model;
}

/* ── Triplet Loss ────────────────────────────────────── */
function tripletLoss(margin = 0.3) {
  return (yTrue, yPred) => {
    // yPred shape: [batch * 3, EMBED_DIM]
    // Reshape: anchor, positive, negative
    const batchSize = yPred.shape[0] / 3;
    const anchors   = yPred.slice([0, 0],          [batchSize, EMBED_DIM]);
    const positives = yPred.slice([batchSize, 0],   [batchSize, EMBED_DIM]);
    const negatives = yPred.slice([batchSize*2, 0], [batchSize, EMBED_DIM]);

    const posDist = tf.sum(tf.square(tf.sub(anchors, positives)), -1);
    const negDist = tf.sum(tf.square(tf.sub(anchors, negatives)), -1);

    const loss = tf.mean(tf.maximum(tf.add(tf.sub(posDist, negDist), margin), 0));
    return loss;
  };
}

/* ── Center Loss (auxiliary for improved clustering) ── */
function centerLoss(centers, lambda = 0.01) {
  return (yTrue, yPred) => {
    // Pull embeddings toward class centers
    const labels = tf.cast(yTrue, 'int32').squeeze();
    const gathered = tf.gather(centers, labels);
    return tf.mul(lambda, tf.mean(tf.sum(tf.square(tf.sub(yPred, gathered)), -1)));
  };
}

/* ── Data Augmentation ───────────────────────────────── */
function augmentFace(imageTensor) {
  return tf.tidy(() => {
    let img = imageTensor;

    // Random horizontal flip
    if (Math.random() > 0.5) {
      img = tf.reverse(img, 1);
    }

    // Random brightness adjustment
    const brightness = tf.randomUniform([], -0.15, 0.15);
    img = tf.clipByValue(tf.add(img, brightness), 0, 1);

    // Random contrast adjustment
    const contrast = tf.randomUniform([], 0.85, 1.15);
    const mean = tf.mean(img);
    img = tf.clipByValue(tf.add(tf.mul(tf.sub(img, mean), contrast), mean), 0, 1);

    // Random noise
    if (Math.random() > 0.6) {
      const noise = tf.randomNormal(img.shape, 0, 0.02);
      img = tf.clipByValue(tf.add(img, noise), 0, 1);
    }

    return img;
  });
}

/* ── Generate Triplets for Training ──────────────────── */
function generateTriplets(embeddingsByUser, batchSize = 16) {
  const userIds = Object.keys(embeddingsByUser).filter(
    uid => embeddingsByUser[uid].length > 0
  );
  if (userIds.length < 2) return null;

  const anchors = [], positives = [], negatives = [];

  for (let i = 0; i < batchSize; i++) {
    // Pick random anchor user
    const aIdx = Math.floor(Math.random() * userIds.length);
    const anchorUser = userIds[aIdx];
    const anchorImages = embeddingsByUser[anchorUser];
    if (anchorImages.length < 2) continue;

    // Anchor and positive from same user
    const shuffled = [...anchorImages].sort(() => Math.random() - 0.5);
    anchors.push(shuffled[0]);
    positives.push(shuffled[1]);

    // Negative from different user
    let nIdx;
    do { nIdx = Math.floor(Math.random() * userIds.length); } while (nIdx === aIdx);
    const negUser = userIds[nIdx];
    const negImages = embeddingsByUser[negUser];
    negatives.push(negImages[Math.floor(Math.random() * negImages.length)]);
  }

  if (anchors.length === 0) return null;
  return { anchors, positives, negatives };
}

/* ── Training Pipeline ───────────────────────────────── */
async function trainEmbeddingModel(model, imagesByUser, options = {}) {
  const {
    epochs = 50,
    batchSize = 16,
    learningRate = 0.001,
    margin = 0.3,
    onEpochEnd = null
  } = options;

  const optimizer = tf.train.adam(learningRate);
  const history = { loss: [], epoch: [] };

  console.log(`[Train] Starting embedding training: ${epochs} epochs, batch=${batchSize}`);
  console.log(`[Train] Users: ${Object.keys(imagesByUser).length}`);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const triplets = generateTriplets(imagesByUser, batchSize);
    if (!triplets) {
      console.warn('[Train] Not enough data for triplets, skipping epoch');
      continue;
    }

    const loss = await tf.tidy(() => {
      // Stack triplet images
      const anchorBatch   = tf.stack(triplets.anchors.map(t =>
        augmentFace(t)));
      const positiveBatch = tf.stack(triplets.positives.map(t =>
        augmentFace(t)));
      const negativeBatch = tf.stack(triplets.negatives.map(t =>
        augmentFace(t)));
      const allInputs = tf.concat([anchorBatch, positiveBatch, negativeBatch], 0);

      // Forward pass + triplet loss
      return optimizer.minimize(() => {
        const embeddings = model.predict(allInputs, { training: true });
        const bs = triplets.anchors.length;
        const anc = embeddings.slice([0, 0], [bs, EMBED_DIM]);
        const pos = embeddings.slice([bs, 0], [bs, EMBED_DIM]);
        const neg = embeddings.slice([bs*2, 0], [bs, EMBED_DIM]);

        const posDist = tf.sum(tf.square(tf.sub(anc, pos)), -1);
        const negDist = tf.sum(tf.square(tf.sub(anc, neg)), -1);
        return tf.mean(tf.maximum(tf.add(tf.sub(posDist, negDist), margin), 0));
      }, true);
    });

    const lossVal = loss ? loss.dataSync()[0] : 0;
    history.loss.push(lossVal);
    history.epoch.push(epoch);

    if (epoch % 5 === 0 || epoch === epochs - 1) {
      console.log(`[Train] Epoch ${epoch+1}/${epochs} — loss: ${lossVal.toFixed(6)}`);
    }
    if (onEpochEnd) await onEpochEnd(epoch, lossVal);
    if (loss) loss.dispose();
  }

  return history;
}

/* ── Extract Embedding from Image Buffer ─────────────── */
async function extractEmbedding(model, imageBuffer) {
  const sharp = require('sharp');

  // Preprocess: resize to INPUT_SIZE, normalize to [0,1]
  const processed = await sharp(imageBuffer)
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: 'cover' })
    .removeAlpha()
    .raw()
    .toBuffer();

  const tensor = tf.tidy(() => {
    const img = tf.tensor3d(new Uint8Array(processed), [INPUT_SIZE, INPUT_SIZE, 3]);
    return img.div(255.0).expandDims(0); // [1, 160, 160, 3]
  });

  const embedding = model.predict(tensor);
  const data = Array.from(await embedding.data());

  tensor.dispose();
  embedding.dispose();

  return data;
}

/* ── Cosine Similarity ───────────────────────────────── */
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

/* ── Euclidean Distance ──────────────────────────────── */
function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/* ── Save / Load Model ───────────────────────────────── */
async function saveModel(model, name = 'embedding_model') {
  const dir = path.join(SAVE_DIR, name);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  await model.save(`file://${dir}`);
  console.log(`[Model] Saved to ${dir}`);
}

async function loadModel(name = 'embedding_model') {
  const dir = path.join(SAVE_DIR, name);
  const jsonPath = path.join(dir, 'model.json');
  if (!fs.existsSync(jsonPath)) return null;
  try {
    const model = await tf.loadLayersModel(`file://${jsonPath}`);
    console.log(`[Model] Loaded from ${dir}`);
    return model;
  } catch (err) {
    console.warn('[Model] Load failed:', err.message);
    return null;
  }
}

module.exports = {
  buildEmbeddingModel,
  trainEmbeddingModel,
  extractEmbedding,
  augmentFace,
  generateTriplets,
  cosineSimilarity,
  euclideanDistance,
  saveModel,
  loadModel,
  tripletLoss,
  INPUT_SIZE,
  EMBED_DIM
};
