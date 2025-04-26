import fs from "fs";
import path from "path";
import sharp from "sharp";
import cliProgress from "cli-progress";

// Каталоги
const datasetDir = path.resolve("dataset");
const outputDir = path.resolve("dataset_processed");

// Размер
const TARGET_WIDTH = 360;
const TARGET_HEIGHT = 380;

// Размер батча — можешь подбирать под возможности системы
const BATCH_SIZE = 50;

async function processAllImages() {
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }

    const allFiles = fs.readdirSync(datasetDir).filter(file => file.endsWith(".jpg"));
    const total = allFiles.length;

    if (total === 0) {
        console.log("✅ Нет изображений для обработки.");
        return;
    }

    const progressBar = new cliProgress.SingleBar({
        format: '📦 Обработка |{bar}| {percentage}% || {value}/{total} файлов',
        barCompleteChar: '\u2588',
        barIncompleteChar: '\u2591',
        hideCursor: true
    }, cliProgress.Presets.shades_classic);

    progressBar.start(total, 0);

    for (let i = 0; i < total; i += BATCH_SIZE) {
        const batch = allFiles.slice(i, i + BATCH_SIZE);

        for (const fileName of batch) {
            const inputPath = path.join(datasetDir, fileName);
            const outputPath = path.join(outputDir, fileName);

            try {
                await sharp(inputPath)
                    .resize(TARGET_WIDTH, TARGET_HEIGHT, { fit: "cover" })
                    .grayscale()
                    .jpeg({ quality: 90 })
                    .toFile(outputPath);
            } catch (err) {
                console.error(`❌ Ошибка при обработке ${fileName}:`, err);
            }

            progressBar.increment();
        }
    }

    progressBar.stop();
    console.log(`🎉 Всё готово! Обработано файлов: ${total}`);
}

processAllImages();
