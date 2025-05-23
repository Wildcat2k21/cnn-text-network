import * as tf from '@tensorflow/tfjs-node-gpu';
import fs from 'fs/promises';
import path from 'path';
import cliProgress from 'cli-progress';

// Разбор аргументов: --model, --teach, --lr
const args = process.argv.slice(2).reduce((acc, cur, i, arr) => {
  if (cur.startsWith('--')) acc[cur.slice(2)] = arr[i+1];
  return acc;
}, {});

const LEARNING_RATE = args.lr ? parseFloat(args.lr) : 0.001;
const MODEL_NAME    = args.model ?? path.basename(args.teach ?? 'unnamed-model');
const TEACH_PATH    = args.teach;

const DATA_DIR     = './converted-images';
const METRIC_DIR   = './metrics-json';
const MODEL_DIR    = './cnn-models';
const TEMP_DIR     = './cnn-temp';

const EPOCHS       = 50;                       // Число эпох для обучения
const BATCH_SIZE   = 32;                       // Размер батча
const IMAGE_HEIGHT = 200;
const IMAGE_WIDTH  = 266;
const OUTPUT_UNITS = 2;                        // Два непрерывных выхода

const SHUFFLE_BUFFER = 2000;

async function createModel() {
  const model = tf.sequential();
  [[32,0.1],[64,0.1],[128,0.5]].forEach(([f,d],i)=>{
    model.add(tf.layers.conv2d({
      inputShape: i===0?[IMAGE_HEIGHT,IMAGE_WIDTH,1]:undefined,
      filters: f, kernelSize:3, padding:'same', useBias:false
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.leakyReLU({alpha:0.1}));
    model.add(tf.layers.maxPooling2d({poolSize:[2,2]}));
    model.add(tf.layers.dropout({rate:d}));
  });
  model.add(tf.layers.globalAveragePooling2d({dataFormat: 'channelsLast'}));
  model.add(tf.layers.dense({units:128}));
  model.add(tf.layers.leakyReLU({alpha:0.1}));
  model.add(tf.layers.dropout({rate:0.3}));
  model.add(tf.layers.dense({units: OUTPUT_UNITS}));           // Линейная регрессия для двух выходов

  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: 'meanSquaredError',                              // MSE для регрессии
    metrics: ['mae']                                       // MAE для обеих метрик
  });
  return model;
}

async function train() {
  let model;
  if (TEACH_PATH) {
    console.log(`💻 Дообучение модели: ${TEACH_PATH}`);
    const json = `file://${path.resolve(TEACH_PATH)}/model.json`;
    model = await tf.loadLayersModel(json);
    model.compile({
      optimizer: tf.train.adam(LEARNING_RATE),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });
  } else {
    console.log(`💻 Создание модели: ${MODEL_NAME}`);
    model = await createModel();
  }

  console.log("📈 Learning rate:", LEARNING_RATE, "\n");
  // Загрузка и подготовка данных
  const files = (await fs.readdir(DATA_DIR)).filter(f => f.endsWith('.jpg'));
  const numSamples = files.length;

  // const files = (await fs.readdir(DATA_DIR)).filter(f=>f.endsWith('.jpg'));
  // const num = files.length;

  async function* gen() {
    for (const f of files) {
      const buf = await fs.readFile(path.join(DATA_DIR,f));
      const m = JSON.parse(await fs.readFile(
        path.join(METRIC_DIR,`${path.basename(f,'.jpg')}.json`),'utf-8'
      ));
      // Используем первые два элемента averageCharMetrics как целевые непрерывные показатели
      yield {buffer: buf, label: [m.averageCharMetrics[4], m.averageLineMetrics[2]]};
    }
  }

  // const trainSize = Math.floor(num * 0.8);
  // const baseDs = tf.data.generator(gen).shuffle(SHUFFLE_BUFFER);
  // const trainDs = baseDs.take(trainSize);
  // const valDs   = baseDs.skip(trainSize);

  // const prep = ds =>
  //   ds
  //     .map(({buffer, label}) => tf.tidy(() => ({
  //       xs: tf.node.decodeImage(buffer,1)
  //            .resizeNearestNeighbor([IMAGE_HEIGHT, IMAGE_WIDTH])
  //            .toFloat().div(255),
  //       ys: tf.tensor1d(label)                                            // 1D тензор [label0, label1]
  //     })))
  //     .batch(BATCH_SIZE)
  //     .prefetch(8);

  // const trainData = prep(trainDs);
  // const valData   = prep(valDs);

  const ds = tf.data.generator(gen).shuffle(numSamples); // фикс имени
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

  const epochBar = new cliProgress.SingleBar({format:'Эпоха {value}/{total}'},cliProgress.Presets.shades_classic);
  epochBar.start(EPOCHS,0);
  let batchBar;
  const callbacks = {
    onEpochBegin: epoch => {
      const totalBatches = Math.ceil(trainSize / BATCH_SIZE);
      batchBar = new cliProgress.SingleBar(
        {format:`Эпоха ${epoch+1} [{bar}] Батч {value}/${totalBatches}`},
        cliProgress.Presets.shades_classic
      );
      batchBar.start(totalBatches,0);
    },
    onBatchEnd: batch => batchBar.update(batch+1),
    onEpochEnd: async (epoch, logs) => {
      batchBar.stop();
      epochBar.increment();
      console.log(` [Эпоха ${epoch+1} — loss=${logs.loss.toFixed(4)}, mae=${logs.mae.toFixed(4)}, `+
                  `val_loss=${logs.val_loss.toFixed(4)}, val_mae=${logs.val_mae.toFixed(4)}]`);
      const out = path.join(TEMP_DIR, MODEL_NAME, `epoch-${epoch+1}`);
      await fs.mkdir(out, {recursive:true});
      await model.save(`file://${out}`);
    }
  };

  await model.fitDataset(trainData, {epochs: EPOCHS, validationData: valData, callbacks, verbose: 0});
  epochBar.stop();
  console.log('✅ Обучение завершено');
  const final = path.join(MODEL_DIR, MODEL_NAME);
  await fs.mkdir(final, {recursive:true});
  await model.save(`file://${final}`);
  console.log(`✅ Сохранено: ${final}`);
}

train().catch(e => { console.error('Error:', e); process.exit(1); });
