import * as tf from '@tensorflow/tfjs-node-gpu';
import fs from 'fs/promises';
import path from 'path';
import cliProgress from 'cli-progress';

// --- Аргументы командной строки ---
const args = process.argv.slice(2).reduce((acc, cur, i, arr) => {
  if (cur.startsWith('--')) acc[cur.slice(2)] = arr[i + 1];
  return acc;
}, {});

const MODEL_NAME    = args.model ?? path.basename(args.teach ?? 'unnamed-model');
const TEACH_PATH    = args.teach;
const LEARNING_RATE = args.lr ? parseFloat(args.lr) : 0.001;

const DATA_DIR      = './converted-images';
const METRIC_DIR    = './metrics-json';
const MODEL_DIR     = './cnn-models';
const TEMP_DIR      = './cnn-temp';

const EPOCHS        = 30;
const BATCH_SIZE    = 32;
const IMAGE_HEIGHT  = 200;
const IMAGE_WIDTH   = 266;
// Ранее было NUM_CLASSES = 14; теперь регрессия одного выхода
const NUM_OUTPUTS   = 1;

// Функция создания свёрточного блока
function convBlock(input, filters, dropRate) {
  let x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', useBias: false }).apply(input);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x);
  x = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x);
  x = tf.layers.dropout({ rate: dropRate }).apply(x);
  return x;
}

// --- Создание модели ---
function createModel() {
  console.log(`\n💻 Создание новой модели: ${MODEL_NAME}`);

  const imgInput = tf.input({ shape: [IMAGE_HEIGHT, IMAGE_WIDTH, 1], name: 'img_input' });
  let x = convBlock(imgInput, 32, 0.1);
  x = convBlock(x, 64, 0.1);
  x = convBlock(x, 128, 0.5);
  x = tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }).apply(x);

  // Новая модель: регрессия одного числа
  const output = tf.layers.dense({ units: NUM_OUTPUTS, activation: 'linear', name: 'output' }).apply(x);

  const model = tf.model({ inputs: imgInput, outputs: output });
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: 'meanSquaredError',
    metrics: ['mse']
  });

  console.log('📈 Learning rate:', LEARNING_RATE);
  return model;
}

// --- Загрузка или создание модели ---
async function prepareModel() {
  if (TEACH_PATH) {
    console.log(`\n💾 Дообучение модели: ${TEACH_PATH}`);
    const modelJson = `file://${path.resolve(TEACH_PATH)}/model.json`;
    const model = await tf.loadLayersModel(modelJson);
    model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError', metrics: ['mse'] });
    return model;
  }
  return createModel();
}

// --- Генератор данных ---
async function* dataGen(files) {
  for (const file of files) {
    const imgBuf = await fs.readFile(path.join(DATA_DIR, file));
    const meta = JSON.parse(await fs.readFile(path.join(METRIC_DIR, `${path.basename(file, '.jpg')}.json`), 'utf-8'));
    
    // Декодирование и нормализация изображения
    const imgTensor = tf.node.decodeImage(imgBuf, 1)
      .resizeNearestNeighbor([IMAGE_HEIGHT, IMAGE_WIDTH])
      .toFloat()
      .div(255);

    // Метрика: средняя метрика символов, 0 индекс
    const labelValue = meta.averageCharMetrics[0];
    const label = tf.scalar(labelValue);

    yield { xs: imgTensor, ys: label };
  }
}

// --- Подготовка датасета ---
async function buildDataset() {
  const files = (await fs.readdir(DATA_DIR)).filter(f => f.endsWith('.jpg'));
  const num = files.length;
  const trainSize = Math.floor(num * 0.8);
  const ds = tf.data.generator(() => dataGen(files)).shuffle(num);
  const trainDs = ds.take(trainSize).batch(BATCH_SIZE).prefetch(1);
  const valDs   = ds.skip(trainSize).batch(BATCH_SIZE).prefetch(1);
  return { trainDs, valDs, trainSize };
}

// --- Обучение ---
async function train() {
  const model = await prepareModel();
  const { trainDs, valDs, trainSize } = await buildDataset();

  const epochBar = new cliProgress.SingleBar({ format: 'Эпоха {value}/{total}', hideCursor: true }, cliProgress.Presets.shades_classic);
  epochBar.start(EPOCHS, 0);
  let batchBar;

  const callbacks = {
    onEpochBegin: epoch => {
      const totalBatches = Math.ceil(trainSize / BATCH_SIZE);
      batchBar = new cliProgress.SingleBar({ format: `Эпоха ${epoch+1} [{bar}] Батч {value}/${totalBatches}`, hideCursor: true }, cliProgress.Presets.shades_classic);
      batchBar.start(totalBatches, 0);
    },
    onBatchEnd: batch => batchBar.update(batch + 1),
    onEpochEnd: async (epoch, logs) => {
      batchBar.stop();
      epochBar.increment();
      console.log(` [Эпоха ${epoch+1} — mse=${logs.mse.toFixed(4)}, val_mse=${logs.val_mse.toFixed(4)}]`);
      const out = path.join(TEMP_DIR, MODEL_NAME, `epoch-${epoch+1}`);
      await fs.mkdir(out, { recursive: true });
      await model.save(`file://${out}`);
    }
  };

  await model.fitDataset(trainDs, { epochs: EPOCHS, validationData: valDs, callbacks, verbose: 0 });
  epochBar.stop();

  console.log('\n✅ Обучение завершено');
  const finalPath = path.join(MODEL_DIR, MODEL_NAME);
  await fs.mkdir(finalPath, { recursive: true });
  await model.save(`file://${finalPath}`);
  console.log(`✅ Модель сохранена в ${finalPath}`);
}

train().catch(e => { console.error('❌ Ошибка во время обучения:', e); process.exit(1); });
