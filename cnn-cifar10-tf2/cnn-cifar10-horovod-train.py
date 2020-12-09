# Example adapted from: https://www.tensorflow.org/tutorials/images/cnn (as of November 19, 2020)
# Original code from web page listed as under the Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)
#
# Ideas for converting original non-disributed code to distributed Horovod come from:
#  https://horovod.readthedocs.io/en/stable/keras.html
#
# Any modifications to the original code are also under the Apache 2.0 License

import tensorflow as tf
import horovod.tensorflow.keras as hvd

from tensorflow.keras import datasets, layers, models


hvd.init()

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

optimizer = tf.optimizers.Adam(0.001 * hvd.size())
hvdOptimizer = hvd.DistributedOptimizer(optimizer)

model.compile(optimizer=hvdOptimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              experimental_run_tf_function=False)

# Broadcast initial variable values from rank 0 to other ranks
cbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0), ]

# Checkpoints only created on rank 0, to avoid race conditions
if hvd.rank() == 0:
    cbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

history = model.fit(train_images, train_labels,
                    steps_per_epoch=1000//hvd.size(), epochs=20, 
                    validation_data=(test_images, test_labels),
                    callbacks=cbacks,
                    verbose=1 if hvd.rank() == 0 else 0)

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
