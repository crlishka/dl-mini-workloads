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

# Load previously saved model
model = tf.keras.models.load_model('./saved-model')

model.summary()

train_loss, train_acc = model.evaluate(train_images,  train_labels, verbose=2)
print('Training set accuracy: ', train_acc)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Test accuracy: ', test_acc)
