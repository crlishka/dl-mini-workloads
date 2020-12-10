# Example adapted from: https://www.tensorflow.org/tutorials/images/cnn (as of November 19, 2020)
# Original code from web page listed as under the Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)
#
# Any modifications to the original code are also under the Apache 2.0 License

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# TensorBoard tracing and profiling
# Profiling requires: "pip install -U tensorboard_plugin_profile"
log_dir = "tf-profile-train"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                      profile_batch=0,
                                                      embeddings_freq=10, write_graph=True)

tf.profiler.experimental.start('tf-profile-train')
history = model.fit(train_images, train_labels, epochs=3, 
                    validation_data=(test_images, test_labels),
                    callbacks=[tensorboard_callback])
tf.profiler.experimental.stop()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
