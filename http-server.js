import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

// —————— Parse CLI args for port и accept origin ———————
const argv = process.argv.slice(2);
let cliPort = null;
let cliAllow = null;
let makeTests = false;
for (let i = 0; i < argv.length; i++) {
  if ((argv[i] === "-p" || argv[i] === "--port") && argv[i+1]) {
    const p = parseInt(argv[i+1], 10);
    if (!isNaN(p)) cliPort = p;
  }

  if ((argv[i] === "-a" || argv[i] === "--allow") && argv[i+1]) {
    const a = parseInt(argv[i+1], 10);
    if (!isNaN(a)) cliAllow = a;
  }

  if ((argv[i] === "-m" || argv[i] === "--make-tests") && argv[i+1]) {
    makeTests = argv[i+1] === "true";
  }
}

// —————— Determine final port ——————
const PORT = cliPort || parseInt(process.env.PORT, 10) || 3015;
const ACCEPT_ORIGIN = cliAllow || process.env.ACCEPT_ORIGIN || "http://localhost:3000";

const app = express();

// Allow CORS from your frontend (or use "*" for any origin)
app.use(cors({
  origin: ACCEPT_ORIGIN,
  methods: ["POST"],
  allowedHeaders: ["Content-Type"]
}));

const imagesDirName = makeTests ? "test-images" : "images-data";
const metricsDirName = makeTests ? "usemodel-json-tests" : "metrics-json";

// Create images-data folder if needed
const datasetDir = path.join(process.cwd(), imagesDirName); //"images-data" "test-images"
if (!fs.existsSync(datasetDir)) fs.mkdirSync(datasetDir, { recursive: true });

// Create metrics-data folder if needed
const metricsDir = path.join(process.cwd(), metricsDirName); //"metrics-json" //"usemodel-json-tests"
if (!fs.existsSync(metricsDir)) fs.mkdirSync(metricsDir, { recursive: true });

// Multer setup: keep files in memory buffer
const storage = multer.memoryStorage();
const upload  = multer({ storage });

// Для тестов
const tempArrForSingleJsonFile = [];

app.post("/images-data", upload.any(), async (req, res) => {
  try {
    // assume fieldnames "images[0]", "images[1]", ... and body.text_metrics = ["7","3",...]
    const images = req.files.filter(f => f.fieldname.startsWith("images["));
    const metrics = Array.isArray(req.body.text_metrics)
      ? req.body.text_metrics
      : [req.body.text_metrics];
    
    images.forEach((file, idx) => {
      // Сохраняем изображения
      const timestamp = Date.now() + idx;
      const filename = `${timestamp}.jpg`;
      fs.writeFileSync(path.join(datasetDir, filename), file.buffer);

      if(makeTests){
        // Для тестов
        const jsonMetricsObj = {name: filename, expects: JSON.parse(metrics[idx])}
        tempArrForSingleJsonFile.push(jsonMetricsObj);
        const jsonStringObj = JSON.stringify(tempArrForSingleJsonFile, null, 2);

        // Сохраняем метрики в один файл тестов
        const metricsFilename = "classifier.json";
        fs.writeFileSync(path.join(metricsDir, metricsFilename), jsonStringObj);
          
        return;
      }

      // Парсим метрики
      const jsonMetricsObj = JSON.parse(metrics[idx]);
      const jsonStringObj = JSON.stringify(jsonMetricsObj, null, 2);

      // Сохраняем метрики
      const metricsFilename = `${timestamp}.json`;
      fs.writeFileSync(path.join(metricsDir, metricsFilename), jsonStringObj);
    });

    res.status(200).json({ success: true, message: "Файлы загружены" });
  } catch (err) {
    console.error("Ошибка при загрузке:", err);
    res.status(500).json({ success: false, error: "Ошибка при обработке данных" });
  }
});

app.listen(PORT, () => {
  console.log(`✅ HTTP-сервер запущен: http://localhost:${PORT}`);
  console.log(`✅ Разрешенный источник: ${ACCEPT_ORIGIN}`);
  
  if(makeTests){
    console.log("🔬 Запущен для сбора тестов");
  }
});
