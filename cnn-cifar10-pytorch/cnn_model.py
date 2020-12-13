# PyTorch CNN CIFAR10 Model, translated from like TF2 Keras model

import torch


# TF2 Keras Model Equivalent:
# 
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 30, 30, 32)        896
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928
# _________________________________________________________________
# flatten (Flatten)            (None, 1024)              0
# _________________________________________________________________
# dense (Dense)                (None, 64)                65600
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                650
# =================================================================
# Total params: 122,570
# Trainable params: 122,570
# Non-trainable params: 0

class CnnModel(torch.nn.Module):

    def __init__(self):

        super(CnnModel, self).__init__()

        self.conv_0 = torch.nn.Conv2d(3, 32, 3)
        self.relu_0 = torch.nn.ReLU()
        self.pool_0 = torch.nn.MaxPool2d(2)
        self.conv_1 = torch.nn.Conv2d(32, 64, 3)
        self.relu_1 = torch.nn.ReLU()
        self.pool_1 = torch.nn.MaxPool2d(2)
        self.conv_2 = torch.nn.Conv2d(64, 64, 3)
        self.relu_2 = torch.nn.ReLU()
        self.dens_0 = torch.nn.Linear(64 * 4 * 4, 64)
        self.relu_3 = torch.nn.ReLU()
        self.dens_1 = torch.nn.Linear(64, 10)

    def forward(self, x):

        #                Keras equivalent: input_shape=(32, 32, 3)
        x = self.relu_0(self.conv_0(x))  # add(Conv2D(32, (3,3), act='relu', ...)
        x = self.pool_0(x)               # add(MaxPooling2D((2, 2))
        x = self.relu_1(self.conv_1(x))  # add(Conv2D(64, (3,3), act='relu', ...)
        x = self.pool_1(x)               # add(MaxPooling2D((2, 2))
        x = self.relu_2(self.conv_2(x))  # add(Conv2D(64, (3,3), act='relu', ...)
        x = x.view(-1, 64 * 4 * 4)       # add(Flatten())
        x = self.relu_3(self.dens_0(x))  # add(Dense(64, acti='relu'))
        x = self.dens_1(x)               # add(Dense(10))
        return x

    def save_filename(self):  return './saved-model.zip'

    def run_inference(self, loader_dataset):

        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader_dataset:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return  100.0 * correct / total  # Accuracy

# End: Class CnnModel
