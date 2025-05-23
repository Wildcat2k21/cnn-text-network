import * as tf from '@tensorflow/tfjs-node-gpu';
import fs from 'fs/promises';
import path from 'path';
import cliProgress from 'cli-progress';

// Разбор аргументов командной строки: --model для имени, --teach для пути к готовой модели, --lr для learning rate
const args = process.argv.slice(2).reduce((acc, cur, i, arr) => {
  if (cur.startsWith('--')) { acc[cur.slice(2)] = arr[i + 1]; }
  return acc;
}, {});

// Гиперпараметр learning rate (по умолчанию 0.001, можно задать через --lr)
const LEARNING_RATE = args.lr ? parseFloat(args.lr) : 0.001;

// Имя модели: либо из --model, либо базовое имя из --teach, либо дефолт
const MODEL_NAME = args.model ?? path.basename(args.teach ?? 'default-model');
const TEACH_PATH = args.teach; // Путь к существующей модели для дообучения

// Пути к данным и директориям
const DATA_DIR   = './converted-images';    // изображения .jpg
const METRIC_DIR = './metrics-json';        // метаданные .json
const MODEL_DIR  = './cnn-models';          // финальные модели
const TEMP_DIR   = './cnn-temp';            // промежуточные сохранения

// Другие гиперпараметры
const EPOCHS       = 30;
const BATCH_SIZE   = 32;
const IMAGE_HEIGHT = 200;
const IMAGE_WIDTH  = 200;
const NUM_CLASSES  = 7;

/**
 * Создание новой модели с указанным LEARNING_RATE
 */
async function createModel() {
  console.log(`\n\n💻 Создание новой модели: ${MODEL_NAME}`);
  const model = tf.sequential();
  [[32, 0.1], [64, 0.1], [128, 0.5]].forEach(([filters, dropRate], i) => {
    model.add(tf.layers.conv2d({
      inputShape: i === 0 ? [IMAGE_HEIGHT, IMAGE_WIDTH, 1] : undefined,
      filters,
      kernelSize: 3,
      padding: 'same',
      useBias: false
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.leakyReLU({ alpha: 0.1 }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.dropout({ rate: dropRate }));
  });
  model.add(tf.layers.globalAveragePooling2d({}));
  model.add(tf.layers.dense({ units: 128 }));
  model.add(tf.layers.leakyReLU({ alpha: 0.1 }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

  // Компиляция с динамическим LEARNING_RATE
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

async function train() {
  let model;

  if (TEACH_PATH) {
    // Загрузка существующей модели и дообучение с новым LEARNING_RATE
    const modelJson = `file://${path.resolve(TEACH_PATH)}/model.json`;
    console.log(`\n\n💾 Загрузка существующей модели из ${TEACH_PATH}`);
    model = await tf.loadLayersModel(modelJson);
    model.compile({
      optimizer: tf.train.adam(LEARNING_RATE),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  } else {
    // Создание новой модели
    model = await createModel();
  }

  console.log("📈 Learning rate:", LEARNING_RATE, "\n");

  // Загрузка и подготовка данных
  const files = (await fs.readdir(DATA_DIR)).filter(f => f.endsWith('.jpg'));
  const numSamples = files.length;

  async function* dataGen() {
    for (const file of files) {
      const buf = await fs.readFile(path.join(DATA_DIR, file));
      const base = path.basename(file, '.jpg');
      const meta = JSON.parse(await fs.readFile(path.join(METRIC_DIR, `${base}.json`), 'utf-8'));
      yield { buffer: buf, label: meta };
    }
  }

  const ds = tf.data.generator(dataGen).shuffle(numSamples);
  const trainSize = Math.floor(numSamples * 0.8);
  const trainDs = ds.take(trainSize);
  const valDs = ds.skip(trainSize);

  const prepare = d => d.map(({ buffer, label }) => tf.tidy(() => ({
    xs: tf.node.decodeImage(buffer, 1)
      .resizeNearestNeighbor([IMAGE_HEIGHT, IMAGE_WIDTH])
      .toFloat().div(255),
    ys: tf.tensor1d(label)
  }))).batch(BATCH_SIZE).prefetch(1);
  const trainData = prepare(trainDs);
  const valData = prepare(valDs);

  // Прогресс-бары по эпохам и батчам
  const epochBar = new cliProgress.SingleBar({ format: 'Эпоха {value}/{total}', hideCursor: true }, cliProgress.Presets.shades_classic);
  epochBar.start(EPOCHS, 0);
  let batchBar;

  const callbacks = {
    onEpochBegin: epoch => {
      const totalBatches = Math.ceil(trainSize / BATCH_SIZE);
      batchBar = new cliProgress.SingleBar({ format: `Эпоха ${epoch+1} / ${EPOCHS} [{bar}] Батч {value}/${totalBatches}`, hideCursor: true }, cliProgress.Presets.shades_classic);
      batchBar.start(totalBatches, 0);
    },
    onBatchEnd: (batch) => batchBar.update(batch + 1),
    onEpochEnd: async (epoch, logs) => {
      batchBar.stop();
      epochBar.increment();
      console.log(` [Эпоха ${epoch+1} — loss=${logs.loss.toFixed(4)}, acc=${(logs.accuracy*100).toFixed(2)}%, val_loss=${logs.val_loss.toFixed(4)}, val_acc=${(logs.val_accuracy*100).toFixed(2)}%]`);
      const out = path.join(TEMP_DIR, MODEL_NAME, `epoch-${epoch+1}`);
      await fs.mkdir(out, { recursive: true });
      await model.save(`file://${out}`);
      console.log(`   Сохранено: эпоха ${epoch+1} → ${out}`);
    }
  };

  // Отключаем стандартные логи tfjs и запускаем обучение
  await model.fitDataset(trainData, {
    epochs: EPOCHS,
    validationData: valData,
    callbacks,
    verbose: 0
  });

  epochBar.stop();
  console.log('\n✅ Обучение завершено');
  const finalPath = path.join(MODEL_DIR, MODEL_NAME);
  await fs.mkdir(finalPath, { recursive: true });
  await model.save(`file://${finalPath}`);
  console.log(`✅ Модель сохранена в ${finalPath}`);
}

train().catch(err => {
  console.error('❌ Ошибка во время обучения:', err);
  process.exit(1);
});
