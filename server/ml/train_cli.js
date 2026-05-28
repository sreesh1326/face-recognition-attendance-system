/* ═══════════════════════════════════════════════════════
   FeD — ml/train_cli.js
   CLI script to trigger model retraining from terminal
   Usage: npm run train
          node ml/train_cli.js [--epochs 50] [--batch 16]
   ═══════════════════════════════════════════════════════ */
require('dotenv').config();
const pipeline = require('./pipeline');

async function main() {
  // Parse CLI args
  const args = process.argv.slice(2);
  const epochs    = parseInt(getArg(args, '--epochs', '50'));
  const batchSize = parseInt(getArg(args, '--batch',  '16'));

  console.log('╔════════════════════════════════════════════╗');
  console.log('║  FeD — Model Training CLI                  ║');
  console.log('╚════════════════════════════════════════════╝');
  console.log(`  Epochs:     ${epochs}`);
  console.log(`  Batch size: ${batchSize}`);
  console.log('');

  try {
    console.log('[CLI] Initializing pipeline...');
    await pipeline.initialize();

    console.log('[CLI] Starting training...');
    const result = await pipeline.retrainModel({ epochs, batchSize });

    if (result.success) {
      console.log('\n══════════════ Training Complete ══════════════');
      console.log(`  Users trained:  ${result.numUsers}`);
      console.log(`  Epochs:         ${result.epochs}`);
      console.log(`  Final loss:     ${result.finalLoss.toFixed(6)}`);
      console.log(`  Duration:       ${(result.durationMs / 1000).toFixed(1)}s`);
      console.log('═══════════════════════════════════════════════\n');
    } else {
      console.warn(`[CLI] Training skipped: ${result.reason}`);
    }
  } catch (err) {
    console.error('[CLI] Training failed:', err.message);
    process.exit(1);
  }

  process.exit(0);
}

function getArg(args, flag, defaultVal) {
  const idx = args.indexOf(flag);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : defaultVal;
}

main();
