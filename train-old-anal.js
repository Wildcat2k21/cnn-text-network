import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs/promises';
import path from 'path';
import cliProgress from 'cli-progress';

// Разбор аргументов командной строки
const args = process.argv.slice(2).reduce((acc, cur, i, arr) => {
  if (cur.startsWith('--')) {
    acc[cur.slice(2)] = arr[i + 1];
  }
  return acc;
}, {});

const MODEL_NAME  = args.model ?? 'default-model';
const DATA_DIR    = './converted-images';
const METRIC_DIR  = './metrics-json';       // Папка с json-файлами метрик
const MODEL_DIR   = './cnn-models';
const EPOCHS      = 10;
const BATCH_SIZE  = 64;
const IMAGE_HEIGHT = 125;
const IMAGE_WIDTH  = 125;
const NUM_OUTPUTS  = 8;                  // 4 метрики символов + 4 метрики строк

function createModel() {
  const model = tf.sequential();

  // Свёрточный блок 1
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    useBias: false
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.leakyReLU({ alpha: 0.1 }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.10 }));

  // Свёрточный блок 2
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', useBias: false }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.leakyReLU({ alpha: 0.1 }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.10 }));

  // Свёрточный блок 3
  model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, padding: 'same', useBias: false }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.leakyReLU({ alpha: 0.1 }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.5 }));

  // Глобальное усреднение вместо Flatten
  model.add(tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }));

  // Полносвязная часть
  model.add(tf.layers.dense({ units: 128 }));
  model.add(tf.layers.leakyReLU({ alpha: 0.1 }));
  model.add(tf.layers.dropout({ rate: 0.30 }));

  // Выходной слой (линейная регрессия)
  model.add(tf.layers.dense({ units: NUM_OUTPUTS }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mse']
  });
  return model;
}

async function train() {
  const files = (await fs.readdir(DATA_DIR)).filter(f => f.endsWith('.jpg'));
  const numSamples   = files.length;
  const totalBatches = Math.ceil(numSamples / BATCH_SIZE);

  async function* bufferGenerator() {
    for (const f of files) {
      const imgBuffer = await fs.readFile(path.join(DATA_DIR, f));
      // Загружаем соответствующий JSON-файл с метриками
      const baseName = path.basename(f, '.jpg');
      const jsonBuffer = await fs.readFile(path.join(METRIC_DIR, `${baseName}.json`), 'utf-8');
      const meta = JSON.parse(jsonBuffer);
      // Собираем в один массив: сначала метрики символов, потом метрики строк
      const labelArray = [
        ...meta.averageCharMetrics,
        ...meta.averageLineMetrics
      ];
      yield { buffer: imgBuffer, label: labelArray };
    }
  }

  const ds = tf.data
    .generator(bufferGenerator)
    .shuffle(numSamples)
    .map(({ buffer, label }) => tf.tidy(() => {
      const img = tf.node
        .decodeImage(buffer, 1)
        .resizeNearestNeighbor([IMAGE_HEIGHT, IMAGE_WIDTH])
        .toFloat()
        .div(255.0);
      const y = tf.tensor1d(label);  // метки регрессии
      return { xs: img, ys: y };
    }))
    .batch(BATCH_SIZE)
    .prefetch(1);

  const model = createModel();

  const epochBar = new cliProgress.SingleBar({ format: 'Эпоха {value}/{total}' }, cliProgress.Presets.shades_classic);
  epochBar.start(EPOCHS, 0);

  let batchBar;
  await model.fitDataset(ds, {
    epochs: EPOCHS,
    callbacks: {
      onEpochBegin: async (epoch) => {
        batchBar = new cliProgress.SingleBar({ format: `Эпоха ${epoch+1} [{bar}] Батч {value}/${totalBatches}` }, cliProgress.Presets.shades_classic);
        batchBar.start(totalBatches, 0);
      },
      onBatchEnd: async (batch) => {
        batchBar.update(batch+1);
      },
      onEpochEnd: async (epoch, logs) => {
        batchBar.stop();
        epochBar.increment();
      
        console.log(` [Эпоха ${epoch+1} — loss=${logs.loss.toFixed(4)}, mse=${logs.mse.toFixed(4)}]`);
      
        // Вывод метрик по каждому выходу
        const preds = model.predict(tf.randomUniform([1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]));
        if (preds.shape[1] === NUM_OUTPUTS) {
          console.log('🧠 Предсказания примерного входа:', Array.from(await preds.data()).map(v => v.toFixed(3)));
        }
      }      
    }
  });

  epochBar.stop();
  console.log('\n✅ Training complete');

  await fs.mkdir(MODEL_DIR, { recursive: true });
  await model.save(`file://${path.join(MODEL_DIR, MODEL_NAME)}`);
  console.log(`✅ Модель сохранена в ${path.join(MODEL_DIR, MODEL_NAME)}`);
}

train().catch(err => {
  console.error(err);
  process.exit(1);
});
