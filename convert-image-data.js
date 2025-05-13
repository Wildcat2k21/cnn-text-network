#!/usr/bin/env node
// convert-images.js

import fs from "fs";
import path from "path";
import sharp from "sharp";
import cliProgress from "cli-progress";

// 1) Разбор аргументов
const args = process.argv.slice(2).reduce((acc, cur, i, arr) => {
  if (cur.startsWith("--")) acc[cur.slice(2)] = arr[i + 1];
  return acc;
}, {});

// 2) Настройка параметров из CLI или по умолчанию
const datasetDir         = path.resolve(args.src      ?? "images-data");
const outputDir          = path.resolve(args.dst      ?? "converted-images");
const TARGET_WIDTH       = parseInt(args.width   ?? "32",  10);
const TARGET_HEIGHT      = parseInt(args.height  ?? "32",  10);
const BATCH_SIZE         = parseInt(args.batch   ?? "10",   10);
const USE_EDGE_DETECTION = args["edge-mode"] === "true";
const BLUR_SIGMA         = args.blur ? parseFloat(args.blur) : null;
const CONTRAST_LEVEL     = args.contrast ? parseFloat(args.contrast) : null;
const BRIGHTNESS_LEVEL   = args.brightness ? parseFloat(args.brightness) : null;
const SHARPEN_SIGMA      = args.sharpen ? parseFloat(args.sharpen) : null;
const USE_GRAYSCALE      = args.gray === "true";
const LIMIT              = args.limit ? parseInt(args.limit, 10) : null;

// 3) Поддержка поворота
const ROTATE_MAP = { "-2": -180, "-1": -90, "1": 90, "2": 180 };
const ROTATE_DEGREES = (args.rotate && ROTATE_MAP[args.rotate] !== undefined)
  ? ROTATE_MAP[args.rotate]
  : null;

// Ядро Sobel X для контуров
const EDGE_KERNEL = { width: 3, height: 3, kernel: [ -1,0,1, -2,0,2, -1,0,1 ] };

async function processAllImages() {
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
  let allFiles = fs.readdirSync(datasetDir)
    .filter(f => f.toLowerCase().endsWith(".jpg"));
  if (LIMIT !== null) allFiles = allFiles.slice(0, LIMIT);
  const total = allFiles.length;
  if (!total) return console.log("✅ Нет изображений для обработки.");

  const progressBar = new cliProgress.SingleBar({
    format: '📦 Обработка |{bar}| {percentage}% || {value}/{total} файлов',
    barCompleteChar: '\u2588', barIncompleteChar: '\u2591', hideCursor: true
  }, cliProgress.Presets.shades_classic);
  progressBar.start(total, 0);

  for (let i = 0; i < total; i += BATCH_SIZE) {
    const batch = allFiles.slice(i, i + BATCH_SIZE);
    for (const fileName of batch) {
      const inputPath  = path.join(datasetDir, fileName);
      const outputPath = path.join(outputDir, fileName);
      try {
        let pipeline = sharp(inputPath)
          .resize(TARGET_WIDTH, TARGET_HEIGHT, { fit: 'cover' });

        if (USE_EDGE_DETECTION) pipeline = pipeline.convolve(EDGE_KERNEL);
        if (BLUR_SIGMA !== null) pipeline = pipeline.blur(BLUR_SIGMA);
        if (CONTRAST_LEVEL !== null) {
          const a = CONTRAST_LEVEL;
          const b = -(0.5 * a) + 0.5;
          pipeline = pipeline.linear(a, b);
        }
        if (BRIGHTNESS_LEVEL !== null) pipeline = pipeline.modulate({ brightness: BRIGHTNESS_LEVEL });
        if (SHARPEN_SIGMA !== null) pipeline = pipeline.sharpen(SHARPEN_SIGMA);
        if (ROTATE_DEGREES !== null) pipeline = pipeline.rotate(ROTATE_DEGREES);
        if (USE_GRAYSCALE) pipeline = pipeline.grayscale();

        await pipeline.jpeg({ quality: 90 }).toFile(outputPath);
      } catch (err) {
        console.error(`❌ Ошибка при обработке ${fileName}:`, err.message);
      }
      progressBar.increment();
    }
  }

  progressBar.stop();
  console.log(`🎉 Всё готово! Обработано файлов: ${total}`);
}

processAllImages().catch(err => {
  console.error("Unexpected error:", err);
  process.exit(1);
});

// Примеры запуска:
// npm run convert -- \
//   --src input-folder --dst output-folder \
//   --width 200 --height 200 --batch 20 \
//   --edge-mode true --blur 2 --contrast 1.5 --brightness 0.8 --sharpen 2 --gray true --rotate -1
// npm run convert -- --src input-folder --dst output-folder --width 200 --height 200 --batch 20 --rotate 2
