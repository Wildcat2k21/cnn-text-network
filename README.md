# Text-Image Generator Training Server

Сервер для приёма, преобразования и обучения на изображениях, сгенерированных с помощью **text-image-generator**.

## Описание проекта

Этот сервис позволяет:
1. Принимать по HTTP POST изображения, сгенерированные text-image-generator, и сохранять их в папку `images-data`.
2. Запускать пакетную обработку (изменение размера, выделение контуров и т.п.) полученных изображений через отдельный скрипт.
3. В дальнейшем использовать подготовленные данные для обучения нейросети.
4. Сохранять обученные модели автоматически в `cnn-models`.
5. Следить за прогрессом обучения, потерями, уровнем ошибки в реальном времени.
6. Проверять обученную модель в режиме прогноза и режиме теста с `expects.json`.

## Установка

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/Wildcat2k21/cnn-text-network.git
   cd cnn-text-network
   ```
2. Установите зависимости:
   ```bash
   npm install
   ```
3. Создайте файл `.env` (по аналогии с `.env.example`), указав порт сервера:
   ```
   PORT=3030
   ```

### Рекомендации
- Используйте версию NodeJS 18.16.1

## Скрипты в `package.json`

```jsonc
"scripts": {
  "serv": "node http-server.js",
  "convert": "node convert-image-data.js",
  "train": "node --max-old-space-size=8192 cnn-train.js",
  "usemodel": "node use-model.js"
}
```

- `npm run serv` — запускает HTTP-сервер (по умолчанию порт из `.env`, или 3030).  
- `npm run convert` — запускает скрипт преобразования изображений.
- `npm run train` — запускает обучение модели на основе подготовленных данных.

## 1. Приём изображений

- **Папка для приёма:** `images-data`
- **Скрипт сервера:** `http-server.js`
- **Запуск сервера:**
  ```bash
  npm run serv
  ```
  — или с переопределением порта:
  ```bash
  npm run serv -- -p 3030 -a 3000
  ```
- **API:**  
  Приём изображений настроен на POST-запрос к эндпоинту:
  ```
  POST /image-data
  ```
  — с бинарным телом запроса (`multipart/form-data` или `application/octet-stream`).  
  Загруженные файлы сохраняются в папку `image-data`.

## 2. Преобразование изображений

Скрипт `convert-image-data.js` позволяет пакетно подготовить изображения:

```bash
npm run convert -- --src images-data --dst converted-images --width 32 --height 32 --batch 20 --edge-mode true --limit 5
```

```bash
npm run convert -- --src test-data --dst conv-test-data --width 200 --height 200 --batch 20
```

### Параметры

- `--src`  
  Путь к исходной папке с изображениями (по умолчанию `text-images`).  
- `--dst`  
  Куда сохранять обработанные файлы (по умолчанию `converted-images`).  
- `--width`  
  Желаемая ширина выходных изображений (в пикселях).  
- `--height`  
  Желаемая высота выходных изображений (в пикселях).  
- `--batch`  
  Размер батча: сколько файлов обрабатывается за одну итерацию (по умолчанию `10`).  
- `--edge-mode`  
  `true` — выполнять выделение контуров (Sobel-фильтр), `false` — без контуров (по умолчанию `false`).  
- `--limit`  
  Ограничение: число первых файлов из `--src`, которые будут обработаны (по умолчанию все).

## 3. Обучение модели

Файл `cnn-train.js` используется для обучения моделей на основе подготовленных данных.

### Настройка параметров модели
В файле `cnn-train.js` можно настроить параметры обучения, такие как:
- Количество эпох (`EPOCHS`)
- Размер батча (`BATCH_SIZE`)
- Размер изображений (`IMAGE_HEIGHT`, `IMAGE_WIDTH`)
- Число классов (`NUM_CLASSES`)

### Запуск обучения
Для запуска обучения с заданным количеством доступной оперативной памяти и числом ядер, нужно отредактировать параметр в `package.json` в разделе `scripts`:

```json
"train": "node --max-old-space-size=8192 cnn-train.js"
```

Здесь `--max-old-space-size=8192` выделяет 8 ГБ ОЗУ для процесса. Для увеличения памяти можно указать больший размер, например, 16 ГБ (8192 — это 8 ГБ).

### Запуск команды для обучения
Чтобы начать обучение, выполните команду с аргументом модели, как показано ниже:

```bash
npm run train -- --model classifier
```

В данном примере модель будет сохранена в папку `cnn-models/hello-world`. Если папки `cnn-models` нет, она будет создана автоматически. Модель будет обучаться и сохраняться в указанной директории.

### Примечание
- Каждую модель можно обучить с уникальным именем, передав его как аргумент. Например, `model hello-world` или любое другое имя, которое будет использоваться для сохранения модели.

## 4. Использование обученной модели

Файл `use-model.js` используется для применения обученной модели для прогнозирования, который можно запустить командой:

```bash
npm run usemodel -- --model cnn-models/mini-upgraded --dir test-images --batch 10 --run-tests usemodel-json-tests/upgrated-mini.json
```

### Параметры

- `--model` — Путь к обученной модели.
- `--dir` — Папка с изображениями для тестирования.
- `--batch` — Размер батча (сколько изображений обрабатывается за одну итерацию).
- `--run-tests` — Указывает, что нужно выполнить тестирование с использованием файла `tests.json`.

### Тестирование с `tests.json`

Файл `tests.json` содержит массив объектов, каждый из которых имеет два поля:
- `name` — имя изображения, которое должно быть протестировано.
- `expects` — ожидаемый результат (может быть числом, если один выход, или массивом, если несколько).

Пример `tests.json`:

```json
[
    { "name": "1745781225690_7.jpg", "expects": 7 },
    { "name": "1745781225692_5.jpg", "expects": 5 },
    { "name": "1745781225693_3.jpg", "expects": 3 },
    { "name": "1745781225694_1.jpg", "expects": 1 },
    { "name": "1745781225695_2.jpg", "expects": 2 }
]
```

Каждый объект в `tests.json` указывает на изображение, для которого ожидается результат. Если в `expects` указано одно число, это предполагаемый класс. Если `expects` — массив, то это несколько возможных классов.


Скрытый (по умолчанию)
node control-generator.js --amount 96465  --interval 5

С UI, чтобы видеть действия в реальном окн
node control-generator.js --amount 96465  --interval 5 --no-headless

```bash
npm run convert -- --src images-data --dst converted-images --width 200 --height 200 --batch 20 --limit 5 --gray true


npm run convert -- --src images-data --dst converted-images --width 200 --height 266 --batch 20 --gray true


npm run convert -- --src test-images --dst conv-test-images --width 200 --height 266 --batch 20 --gray true


npm run convert -- --src test-images --dst conv-test-images --width 200 --height 200 --batch 20 --gray true

npm run convert -- --src real-images --dst conv-real-images --width 200 --height 200 --batch 20 --limit 5 --gray true

npm run convert -- --src real-images --dst conv-real-images --width 200 --height 200 --batch 20 --gray true

npm run convert -- --src test-images --dst conv-test-images --width 200 --height 200 --batch 20 --gray true
```


```bash
node use-model.js --model cnn-temp/font-new/epoch-0 --imgdir conv-test-images --batch 10 --run-tests usemodel-json-tests/classifier.json

node use-model.js --model cnn-temp/epoch-1 --imgdir conv-test-images --batch 10 --run-tests usemodel-json-tests/classifier.json

node use-model.js --model cnn-temp/classifier/epoch-4 --imgdir conv-test-images --batch 10 --run-tests usemodel-json-tests/classifier.json

node use-model.js --model cnn-temp/classifier/epoch-4 --imgdir conv-real-images --batch 10

node use-model.js --model cnn-temp/epoch-4/epoch-2 --imgdir conv-test-images --batch 10 --run-tests usemodel-json-tests/classifier.json

node use-model.js --model cnn-temp/epoch-4/epoch-2 --imgdir conv-real-images --batch 10 --run-tests usemodel-json-tests/classifier.json

node use-model.js --model cnn-temp/charWH/epoch-1 --imgdir conv-real-images --batch 10 --run-tests usemodel-json-tests/classifier.json
```

✅ Да
✅ Нет

```bash
npm run convert -- --src test-data --dst conv-test-data --width 200 --height 200 \
  --batch 20 --edge-mode true --blur 2 --contrast 1.5 --brightness 1.2 --rotate -1 (1 -2 2)
```

npm run convert -- --src real-images --dst conv-real-images --width 200 --height 200 \
  --batch 20 --contrast 1.5 --brightness -1.2


  npm run convert -- --src test-data --dst conv-test-data \
  --width 200 --height 200 --batch 20 \
  --edge-mode true --blur 2 --gray true --contrast 1.5 --brightness 0.8 --sharpen 2

  npm run convert -- --src real-images --dst conv-real-images \
  --width 200 --height 200 --batch 20 --sharpen 2




node control-generator.js -a 118990 -b 100 -i 5

node control-generator.js -a 118990 -b 1000 -i 5 --no-headless

node control-generator.js -a 200000 -b 3000 -i 5 --no-headless




control-generator.js -a 10000 -b 500 -i 5     # headful
control-generator.js -a 10000 -b 500 -i 5 -H  # headless



node control-generator.js \
  --amount 10000 \
  --batch 500 \
  --interval 5 \
  --headless



node control-generator.js \
  --amount 10000 \
  --batch 500 \
  --interval 5



💻 Model: classifier
Эпоха 0/30Epoch 1 / 30
eta=0.0 =====================================================================> 
Эпоха 1/30 [Эпоха 1 — loss=0.2324, acc=NaN%, val_loss=0.2383, val_acc=NaN%]
Epoch 2 / 30
eta=0.0 =====================================================================> 
Эпоха 2/30 [Эпоха 2 — loss=0.1023, acc=NaN%, val_loss=0.4677, val_acc=NaN%]
Epoch 3 / 30
eta=0.0 =====================================================================> 
Эпоха 3/30 [Эпоха 3 — loss=0.0798, acc=NaN%, val_loss=0.3159, val_acc=NaN%]
Epoch 4 / 30
eta=0.0 =====================================================================> 
Эпоха 4/30 [Эпоха 4 — loss=0.0635, acc=NaN%, val_loss=0.0881, val_acc=NaN%]
Epoch 5 / 30
eta=0.0 =====================================================================> 
Эпоха 5/30 [Эпоха 5 — loss=0.0556, acc=NaN%, val_loss=5.1976, val_acc=NaN%]
Epoch 6 / 30
Эпоха 6 [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] Батч 179/3750^C


Дообучение
💾 Загрузка существующей модели из cnn-temp/classifier/epoch-4
📈 Learning rate: 0.001 

Эпоха 1 / 30 [████████████████████████████████████████] Батч 3750/3750
Эпоха 1/30 [Эпоха 1 — loss=0.0207, acc=NaN%, val_loss=0.3227, val_acc=NaN%]
   Сохранено: эпоха 1 → cnn-temp/epoch-4/epoch-1
Эпоха 2 / 30 [████████████████████████████████████████] Батч 3750/3750
Эпоха 2/30 [Эпоха 2 — loss=0.0146, acc=NaN%, val_loss=0.0155, val_acc=NaN%]
   Сохранено: эпоха 2 → cnn-temp/epoch-4/epoch-2
Эпоха 3 / 30 [████████████████████████████████████████] Батч 3750/3750
Эпоха 3/30 [Эпоха 3 — loss=0.0132, acc=NaN%, val_loss=0.0329, val_acc=NaN%]
   Сохранено: эпоха 3 → cnn-temp/epoch-4/epoch-3
Эпоха 4 / 30 [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] Батч 92/3750^C


npm run train -- --model charWH

npm run train -- --teach cnn-temp/font-classifier/epoch-1

npm run train -- --teach cnn-temp/epoch-1/epoch-1
