import * as tf from '@tensorflow/tfjs-node-gpu';

console.log('TensorFlow.js version:', tf.version.tfjs);
console.log('CUDA version:', tf.versions?.cuda || 'unknown');
console.log('cuDNN version:', tf.versions?.cudnn || 'unknown');

const isGPU = tf.engine().backendInstance?.isUsingGpuDevice ?? false;
console.log('GPU доступен:', isGPU);

// Проверка простой математики на тензорах
const a = tf.tensor([1, 2, 3]);
const b = tf.tensor([4, 5, 6]);
const c = a.add(b);
c.print(); // [5, 7, 9]
