import * as tf from '@tensorflow/tfjs-node-gpu';

console.log("Backends:", tf.getBackend());  // Должен быть 'tensorflow'

tf.tensor([1, 2, 3, 4]).square().print();  // Убедись, что вычисления работают

console.log("Is GPU available:", tf.engine().backendInstance.isUsingGpuDevice);
