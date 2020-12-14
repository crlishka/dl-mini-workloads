#!  /usr/bin/env python

# Example adapted from TF tutorials at: https://www.tensorflow.org/tutorials/keras/classification
# Original code from web page listed as under the Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)
#
# Any modifications to the original code are also under the Apache 2.0 License

import datetime
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Put into NHWC form, by adding channel=1 to end
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([  # Sequential densenet model
  tf.keras.layers.Flatten(input_shape=input_shape),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
model.summary()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# TensorBoard tracing and profiling (for steps 500-504)
# Profiling requires: "pip install -U tensorboard_plugin_profile"
log_dir = "tf-profile-train"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                         histogram_freq=1, profile_batch='500,504',
                         embeddings_freq=10, write_graph=True)

# Train -- adjust model parameters (weights) to minimize the loss
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('Test accuracy: ', test_acc)  # Evaluate model against test set

model.save('./saved-model')  # Save model for inference runs
