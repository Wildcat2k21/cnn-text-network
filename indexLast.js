// import express from "express";
// import multer from "multer";
// import fs from "fs";
// import path from "path";
// import cors from "cors";
// import dotenv from "dotenv";
// dotenv.config();

// const app = express();
// const PORT = process.env.PORT || 3001;

// // Позволить CORS запросы от любого источника (или укажи свой)
// app.use(cors({
//     origin: "http://localhost:5173", // или "*" для всех, если не боишься
//     methods: ["POST"],
//     allowedHeaders: ["Content-Type"]
// }));

// app.use(cors());

// // Создаем папку, если не существует
// const datasetDir = path.join(process.cwd(), "dataset");
// if (!fs.existsSync(datasetDir)) fs.mkdirSync(datasetDir);

// // Конфигурация multer — храним файлы во временной папке
// const storage = multer.memoryStorage(); // чтобы обработать вручную
// const upload = multer({ storage });

// app.post("/dataset", upload.any(), async (req, res) => {
//     try {
//         const images = req.files.filter(file => file.fieldname.startsWith("images["));
//         const metrics = req.body;

//         console.log(metrics);

//         images.forEach((file, index) => {
//             const timestamp = Date.now() + index; // чтобы не затирались

//             const filePath = path.join(datasetDir, `${timestamp}.jpg`);
//             fs.writeFileSync(filePath, file.buffer);

//             // Если нужны метрики — сохраняем как JSON
//             const metricField = `text_metrics[${index}]`;
//             if (metrics[metricField]) {
//                 const metricPath = path.join(datasetDir, `${timestamp}.json`);
//                 fs.writeFileSync(metricPath, metrics[metricField]);
//             }
//         });

//         res.status(200).json({ success: true, message: "Файлы загружены" });
//     } catch (err) {
//         console.error("Ошибка при загрузке:", err);
//         res.status(500).json({ success: false, error: "Ошибка при обработке данных" });
//     }
// });

// app.listen(PORT, () => {
//     console.log(`Сервер слушает на http://localhost:${PORT}`);
// });
