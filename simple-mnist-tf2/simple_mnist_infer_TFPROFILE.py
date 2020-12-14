#!  /usr/bin/env python

# Example adapted from TF tutorials at: https://www.tensorflow.org/tutorials/keras/classification
# Original code from web page listed as under the Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)
#
# Any modifications to the original code are also under the Apache 2.0 License

import datetime
import tensorflow as tf
import numpy

def  run_inference(model, name, input_images, input_labels, callbacks=None):
  predictions = model.predict(input_images, batch_size=100, callbacks=callbacks)
  matches = 0
  for idx in range(0, len(input_images)):
    predicted = numpy.argmax(predictions[idx])
    if int(predicted) == int(input_labels[idx]):  matches += 1
  accuracy = float(matches) / float(len(input_images))
  print('Accuracy on %d %s images: %f' % (len(input_images), name, accuracy))
# End def run_inference(...)

mnist = tf.keras.datasets.mnist  # Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Put into NHWC form, by adding channel=1 to end, input_shape = (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.load_model('./saved-model') # Load previously saved model
model.summary() ; model.compile()

log_dir = "tf-profile-infer"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                         histogram_freq=1, profile_batch='500,504',
                         embeddings_freq=10, write_graph=True)

with tf.profiler.experimental.Profile(log_dir):  # Profile one test run
  run_inference(model, 'test', x_test, y_test, tensorboard_callback)
# Artificially run more inference, to get reasonable sample for timing
for i in range(1, 50):   # Run for over a minute on targeted NUC hardware
  run_inference(model, 'test', x_test, y_test)
  run_inference(model, 'train', x_train, y_train)
