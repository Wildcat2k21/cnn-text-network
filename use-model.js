import fs from 'fs/promises';
import path from 'path';
import { Command } from 'commander';

// Constants for input image dimensions
const INPUT_WIDTH = 200;
const INPUT_HEIGHT = 266;

// Utility: generic evaluation of model outputs against expected values
function evaluateOutputs(outputs, expects = null) {
  // outputs: Array of numbers [0..1], expects: Array of 0/1 or null
  const results = outputs.map((score, idx) => {
    const conf = score * 100;
    const actual = score >= 0.5 ? 1 : 0;
    const expected = expects ? expects[idx] : null;
    const pass = expected === null ? null : (actual === expected);
    return { category: idx + 1, actual, expected, pass, confidence: conf };
  });
  const overall = expects
    ? results.every(r => r.pass)
    : null;
  return { results, overall };
}

async function loadTests(testFile) {
  const raw = await fs.readFile(testFile, 'utf-8');
  const tests = JSON.parse(raw);
  if (!Array.isArray(tests)) {
    throw new Error('JSON должен быть массивом объектов { name, expects }');
  }
  return tests;
}

async function main() {
  const program = new Command();
  program
    .requiredOption('-m, --model <path>', 'path to the trained model')
    .requiredOption('-i, --imgdir <path>', 'directory with test images')
    .option('-b, --batch <number>', 'batch size', '1')
    .option('-r, --run-tests <path>', 'JSON file with expected results');

  program.parse(process.argv);
  const opts = program.opts();
  const modelDir = opts.model;
  const imageDir = opts.imgdir;
  const batchSize = parseInt(opts.batch, 10) || 1;
  const testFile = opts.runTests || null;

  let tests = null;
  if (testFile) {
    try {
      tests = await loadTests(testFile);
    } catch (err) {
      console.error('Ошибка чтения тестов:', err.message);
      process.exit(1);
    }
  }

  // Load TensorFlow model dynamically
  const tf = await import('@tensorflow/tfjs-node');
  const modelPath = `file://${path.resolve(modelDir, 'model.json')}`;
  const model = await tf.loadLayersModel(modelPath);
  console.log(`✅ Модель загружена: ${modelPath}`);
  console.log(`ℹ️ Модель ожидает вход: ${INPUT_WIDTH}x${INPUT_HEIGHT} (1 канал)`);

  // Read image files
  const files = (await fs.readdir(imageDir))
    .filter(f => /\.(jpe?g|png)$/i.test(f));
  if (!files.length) {
    console.error('В папке нет изображений .jpg/.png');
    process.exit(1);
  }

  let total = 0;
  let passed = 0;

  for (let i = 0; i < files.length; i += batchSize) {
    const batch = files.slice(i, i + batchSize);

    // Load and preprocess batch
    const tensors = await Promise.all(batch.map(async file => {
      const buf = await fs.readFile(path.join(imageDir, file));
      return tf.tidy(() =>
        tf.node.decodeImage(buf, 1)
          .resizeNearestNeighbor([INPUT_HEIGHT, INPUT_WIDTH])
          .toFloat()
          .div(255.0)
          .expandDims(0)
      );
    }));
    const input = tf.concat(tensors, 0);
    const preds = model.predict(input);
    const arr = await preds.array();

    // Iterate results
    for (let j = 0; j < batch.length; j++) {
      const file = batch[j];
      const scores = arr[j];
      //fontCatOutput - ключ в expects
      const expects = tests ? (tests.find(t => t.name === file)?.expects.fontCatOutput ?? null) : null;
      const { results, overall } = evaluateOutputs(scores, expects);

      console.log(`\nИзображение: "${file}":`);
      results.forEach(r => {
        const mark = r.actual === 1 ? '✅' : '❌';
        const label = r.actual === 1 ? 'Да' : 'Нет';
        const pct = r.confidence.toFixed(2);
        let line = `    Категория ${r.category}: ${mark} ${label} (${pct}%)`;
        if (tests && r.expected !== null && r.pass === false) {
          line += ` - Ожидалось "${r.expected === 1 ? 'Да' : 'Нет'}"`;
        }
        console.log(line);
      });

      if (tests) {
        console.log(`\n    Заключение: Тест ${overall ? 'пройден ✅' : 'провален ❌'}\n`);
        total++;
        if (overall) passed++;
      }
    }

    tf.dispose([input, preds, ...tensors]);
  }

  // Summary
  console.log('✅ Тесты завершены');
  if (tests) {
    const pct = total ? ((passed / total) * 100).toFixed(2) : '0.00';
    if (passed === total) {
      console.log('Все тесты прошли успешно!');
    } else {
      console.log(`⚠️ Не все тесты были пройдены (${passed}/${total}, ${pct}%)`);
    }
  }
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});

// Примеры запуска:
// Режим обозрения:
// npm run usemodel -- --model cnn-temp/classifier/epoch-1 --imgdir conv-test-data --batch 10
// Режим тестирования:
// npm run usemodel -- --model cnn-temp/classifier/epoch-1 --imgdir conv-test-data --batch 10 --run-tests usemodel-json-tests/classifier.json
