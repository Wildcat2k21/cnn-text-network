// check-tf.js
const tf = require('@tensorflow/tfjs-node');

console.log('✅ tfjs-node version:', tf.version.tfjs);
console.log('✅ Backend in use: ', tf.getBackend());

// Optional: create and print a tiny tensor to prove the binary is working
tf.tensor([42, 99]).print();
