#!/usr/bin/env node

import puppeteer from 'puppeteer';
import { Command } from 'commander';
import cliProgress from 'cli-progress';

(async () => {
  const program = new Command();
  program
    .name('control-generator')
    .description('Управление генератором изображений через Puppeteer')
    .version('1.2.1')
    .requiredOption('-a, --amount <number>', 'общее число изображений для генерации', value => parseInt(value, 10))
    .option('-b, --batch <number>', 'число изображений за один батч', value => parseInt(value, 10), 100)
    .option('-i, --interval <seconds>', 'интервал опроса прогресса (с)', value => parseFloat(value), 5)
    .option('--no-headless', 'открыть браузер с UI вместо headless')
    .parse(process.argv);

  const { amount, batch: batchOpt, interval, headless } = program.opts();
  const url = 'http://localhost:3000/';

  const browser = await puppeteer.launch({
    headless,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
    defaultViewport: null, // Дополнительно можно настроить размер окна
    protocolTimeout: 3600 * 1000 // Увеличение времени таймаута
  });

  let totalGenerated = 0;
  const bar = new cliProgress.SingleBar({
    format: 'Progress |{bar}| {percentage}% | {value}/{total} imgs | Rate: {rate}/s',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: true
  });
  bar.start(amount, 0, { rate: '0.00' });

  while (totalGenerated < amount) {
    const remaining = amount - totalGenerated;
    const batchSize = Math.min(batchOpt, remaining);
    console.log(`Starting new batch: batchSize=${batchSize}, totalGenerated=${totalGenerated}`);

    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle0' });

    await page.waitForFunction(
      'typeof window.setRenderAmount === "function" && typeof window.startRender === "function"',
      { timeout: 3600 * 1000 }
    );

    // Сбросим счётчик
    await page.evaluate(() => { window.renderCount = 0; });

    // Установим batchSize и сразу вернём реальное значение renderAmount
    const actual = await page.evaluate(size => {
      window.setRenderAmount(size);
      return window.renderAmount;
    }, batchSize);
    console.log(`Requested size=${batchSize}, page renderAmount=${actual}`);
    if (actual !== batchSize) {
      console.error('Ошибка: renderAmount не соответствует batchSize!');
      await browser.close();
      process.exit(1);
    }

    // Запускаем рендер
    await page.evaluate(() => window.startRender());

    // Слежение за прогрессом
    let prevCount = 0;
    const intervalMs = interval * 1000;
    while (true) {
      await new Promise(r => setTimeout(r, intervalMs));
      const count = await page.evaluate(() => window.renderCount);
      const delta = count - prevCount;
      prevCount = count;
      totalGenerated += delta;
      const rate = (delta / interval).toFixed(2);
      bar.update(totalGenerated, { rate });
      if (count >= batchSize) break;
    }

    // Завершение батча
    await page.evaluate(() => window.stopRender && window.stopRender());
    await page.close();
  }

  bar.stop();
  console.log('✅ Генерация всех изображений завершена.');
  await browser.close();
  process.exit(0);
})();
