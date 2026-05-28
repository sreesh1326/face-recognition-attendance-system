/* ═══════════════════════════════════════════════════════
   FeD — ml/face_detector.js
   Custom Face Detection CNN — SSD-MobileNet-inspired
   Detects face bounding boxes in images without any
   pre-built library. Trained from scratch on face data.
   ═══════════════════════════════════════════════════════ */
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

const DET_SIZE = 320;

/* ── Depthwise Separable Conv Block ──────────────────── */
function dsSepConvBlock(input, filters, stride = 1, name = '') {
  let x = tf.layers.depthwiseConv2d({
    kernelSize: 3, strides: stride, padding: 'same',
    useBias: false, name: `${name}_dw`
  }).apply(input);
  x = tf.layers.batchNormalization({ name: `${name}_dw_bn` }).apply(x);
  x = tf.layers.activation({ activation: 'relu6' }).apply(x);

  x = tf.layers.conv2d({
    filters, kernelSize: 1, strides: 1,
    padding: 'same', useBias: false, name: `${name}_pw`
  }).apply(x);
  x = tf.layers.batchNormalization({ name: `${name}_pw_bn` }).apply(x);
  x = tf.layers.activation({ activation: 'relu6' }).apply(x);
  return x;
}

/* ── Build Face Detection Model ──────────────────────── */
function buildDetectorModel() {
  const input = tf.input({ shape: [DET_SIZE, DET_SIZE, 3], name: 'detector_input' });

  // MobileNet-style backbone
  let x = tf.layers.conv2d({
    filters: 32, kernelSize: 3, strides: 2,
    padding: 'same', useBias: false, name: 'conv_stem'
  }).apply(input);
  x = tf.layers.batchNormalization({ name: 'stem_bn' }).apply(x);
  x = tf.layers.activation({ activation: 'relu6' }).apply(x);

  x = dsSepConvBlock(x, 64,  1, 'block1');
  x = dsSepConvBlock(x, 128, 2, 'block2');
  x = dsSepConvBlock(x, 128, 1, 'block3');
  x = dsSepConvBlock(x, 256, 2, 'block4');
  x = dsSepConvBlock(x, 256, 1, 'block5');
  x = dsSepConvBlock(x, 512, 2, 'block6');
  // 4 repeated blocks at 512
  x = dsSepConvBlock(x, 512, 1, 'block7');
  x = dsSepConvBlock(x, 512, 1, 'block8');
  x = dsSepConvBlock(x, 512, 1, 'block9');

  // Detection head
  x = tf.layers.globalAveragePooling2d({ name: 'det_gap' }).apply(x);
  x = tf.layers.dense({ units: 256, activation: 'relu', name: 'det_fc1' }).apply(x);
  x = tf.layers.dropout({ rate: 0.3 }).apply(x);

  // Output: [confidence, x, y, w, h] normalized 0..1
  const detOutput = tf.layers.dense({
    units: 5, activation: 'sigmoid', name: 'det_output'
  }).apply(x);

  const model = tf.model({ inputs: input, outputs: detOutput, name: 'FaceDetectorNet' });

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: detectionLoss
  });

  return model;
}

/* ── Detection Loss (confidence + bbox regression) ──── */
function detectionLoss(yTrue, yPred) {
  // yTrue/yPred shape: [batch, 5] => [conf, x, y, w, h]
  const trueConf = yTrue.slice([0, 0], [-1, 1]);
  const predConf = yPred.slice([0, 0], [-1, 1]);
  const trueBbox = yTrue.slice([0, 1], [-1, 4]);
  const predBbox = yPred.slice([0, 1], [-1, 4]);

  // Binary cross-entropy for confidence
  const confLoss = tf.losses.sigmoidCrossEntropy(trueConf, predConf);

  // Smooth L1 for bbox regression (only for positive samples)
  const mask = trueConf;
  const diff = tf.abs(tf.sub(trueBbox, predBbox));
  const smoothL1 = tf.where(
    tf.less(diff, 1.0),
    tf.mul(0.5, tf.square(diff)),
    tf.sub(diff, 0.5)
  );
  const bboxLoss = tf.mean(tf.mul(mask, tf.sum(smoothL1, -1, true)));

  return tf.add(confLoss, tf.mul(2.0, bboxLoss));
}

/* ── Train Face Detector ─────────────────────────────── */
async function trainDetector(model, faceImages, nonFaceImages, options = {}) {
  const { epochs = 30, batchSize = 8, learningRate = 0.001 } = options;
  const optimizer = tf.train.adam(learningRate);

  console.log(`[DetTrain] Training detector: ${faceImages.length} faces, ${nonFaceImages.length} non-faces`);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const loss = tf.tidy(() => {
      const batch = [];
      const labels = [];

      // Positive samples (faces with bbox labels)
      for (let i = 0; i < Math.min(batchSize / 2, faceImages.length); i++) {
        const idx = Math.floor(Math.random() * faceImages.length);
        batch.push(faceImages[idx].image);
        labels.push(faceImages[idx].label); // [1, x, y, w, h]
      }

      // Negative samples (no face)
      for (let i = 0; i < Math.min(batchSize / 2, nonFaceImages.length); i++) {
        const idx = Math.floor(Math.random() * nonFaceImages.length);
        batch.push(nonFaceImages[idx]);
        labels.push([0, 0, 0, 0, 0]);
      }

      if (batch.length === 0) return tf.scalar(0);

      const inputBatch = tf.stack(batch);
      const labelBatch = tf.tensor2d(labels);

      return optimizer.minimize(() => {
        const pred = model.predict(inputBatch, { training: true });
        return detectionLoss(labelBatch, pred);
      }, true);
    });

    const lossVal = loss ? loss.dataSync()[0] : 0;
    if (epoch % 5 === 0) {
      console.log(`[DetTrain] Epoch ${epoch+1}/${epochs} — loss: ${lossVal.toFixed(6)}`);
    }
    if (loss) loss.dispose();
  }
}

/* ── Detect Faces in Image Buffer ────────────────────── */
async function detectFaces(model, imageBuffer) {
  const processed = await sharp(imageBuffer)
    .resize(DET_SIZE, DET_SIZE, { fit: 'cover' })
    .removeAlpha()
    .raw()
    .toBuffer();

  const { width: origW, height: origH } = await sharp(imageBuffer).metadata();

  const tensor = tf.tidy(() => {
    return tf.tensor3d(new Uint8Array(processed), [DET_SIZE, DET_SIZE, 3])
      .div(255.0).expandDims(0);
  });

  const prediction = model.predict(tensor);
  const [conf, x, y, w, h] = await prediction.data();

  tensor.dispose();
  prediction.dispose();

  if (conf < 0.5) return [];

  return [{
    confidence: conf,
    box: {
      x: Math.round(x * origW),
      y: Math.round(y * origH),
      width:  Math.round(w * origW),
      height: Math.round(h * origH)
    }
  }];
}

/* ── Crop Face from Image ────────────────────────────── */
async function cropFace(imageBuffer, box) {
  const { width: imgW, height: imgH } = await sharp(imageBuffer).metadata();

  // Clamp box to image bounds with padding
  const pad = 20;
  const left   = Math.max(0, box.x - pad);
  const top    = Math.max(0, box.y - pad);
  const right  = Math.min(imgW, box.x + box.width + pad);
  const bottom = Math.min(imgH, box.y + box.height + pad);

  return sharp(imageBuffer)
    .extract({
      left, top,
      width:  right - left,
      height: bottom - top
    })
    .resize(160, 160, { fit: 'cover' })
    .toBuffer();
}

module.exports = {
  buildDetectorModel,
  trainDetector,
  detectFaces,
  cropFace,
  detectionLoss,
  DET_SIZE
};
