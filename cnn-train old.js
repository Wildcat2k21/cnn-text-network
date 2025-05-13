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

// Получение названия модели (если оно передано, иначе использовать дефолтное имя)
const MODEL_NAME = args.model ?? 'default-model';
const DATA_DIR     = './converted-images';
const MODEL_DIR    = './cnn-models';  // Папка, куда будем сохранять модель
const EPOCHS       = 10;
const BATCH_SIZE   = 200;
const IMAGE_HEIGHT = 32;
const IMAGE_WIDTH  = 32;
const NUM_CLASSES  = 10;

function createModel() {
  const model = tf.sequential();

  // — первый свёрточный блок (без изменений) —
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    useBias: false
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.10 }));

  // — второй свёрточный блок —
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    padding: 'same',
    useBias: false
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.10 }));

  // — третий свёрточный блок —
  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    padding: 'same',
    useBias: false
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.5 }));

  // — глобальное усреднение вместо Flatten —
  model.add(tf.layers.globalAveragePooling2d({}));

  // — «голова» сети —
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.30 }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

// — остальной код train() без изменений, только MODEL_DIR поправить на './cnn-models'

async function train() {
  const files = (await fs.readdir(DATA_DIR)).filter(f => f.endsWith('.jpg'));
  const numSamples   = files.length;
  const totalBatches = Math.ceil(numSamples / BATCH_SIZE);

  async function* bufferGenerator() {
    for (const f of files) {
      const buffer = await fs.readFile(path.join(DATA_DIR, f));
      const label  = parseInt(f.split('_')[1], 10);
      yield { buffer, label };
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
      const y = tf.oneHot(label, NUM_CLASSES);
      return { xs: img, ys: y };
    }))
    .batch(BATCH_SIZE)
    .prefetch(1);

  const model = createModel();

  const epochBar = new cliProgress.SingleBar({
    format: 'Эпоха {value}/{total}'
  }, cliProgress.Presets.shades_classic);
  epochBar.start(EPOCHS, 0);

  let batchBar;
  await model.fitDataset(ds, {
    epochs: EPOCHS,
    callbacks: {
      onEpochBegin: async (epoch) => {
        batchBar = new cliProgress.SingleBar({
          format: `Эпоха ${epoch + 1} [{bar}] Батч {value}/${totalBatches} | ETA: {eta_formatted}`
        }, cliProgress.Presets.shades_classic);
        batchBar.start(totalBatches, 0);
      },
      onBatchEnd: async (batch) => {
        batchBar.update(batch + 1);
      },
      onEpochEnd: async (epoch, logs) => {
        batchBar.stop();
        epochBar.increment();
        const mem = tf.memory();
        console.log(
          `\n[Эпоха ${epoch + 1} — потеря=${logs.loss.toFixed(4)}, ` +
          `точность=${(logs.acc * 100).toFixed(2)}% | ` +
          `тензоров=${mem.numTensors}, байт=${mem.numBytes}]`
        );
      }
    }
  });

  epochBar.stop();
  console.log('\n✅ Training complete');

  // Создаём папку cnn-models, если она не существует
  await fs.mkdir(path.join(MODEL_DIR), { recursive: true });

  // Сохраняем модель в указанную папку с именем, полученным от аргумента model
  await model.save(`file://${path.join(MODEL_DIR, MODEL_NAME)}`);
  console.log(`✅ Модель сохранена в ${path.join(MODEL_DIR, MODEL_NAME)}`);
}

train().catch(err => {
  console.error(err);
  process.exit(1);
});
