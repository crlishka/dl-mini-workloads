# PyTorch CNN CIFAR10 Inference Model

import os.path

import torch
import torchvision as tv
import cnn_model


transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])

# CIFAR10 labels: ['plane', 'car', 'bird', 'cat', 'deer',
#                  'dog', 'frog', 'horse', 'ship', 'truck']
train = tv.datasets.CIFAR10(root='./cifar10-data', transform=transform,
                            download=True, train=True)
test  = tv.datasets.CIFAR10(root='./cifar10-data', transform=transform,
                            download=True, train=False)

batch_size = 32
loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                           num_workers=2, shuffle=True)
loader_test  = torch.utils.data.DataLoader(test,  batch_size=batch_size,
                                           num_workers=2, shuffle=False)

model = cnn_model.CnnModel()

if not os.path.isfile(model.save_filename()):
    print('ERROR: you must run the cnn-cifar10-train.py script to create the saved model')

else:

    model.load_state_dict(torch.load(model.save_filename()))

    print('training-set  size: %d images  accuracy: %5.2f%%' %
          (len(train.targets), model.run_inference(loader_train)))
    print('test-set      size: %d images  accuracy: %5.2f%%' %
          (len(test.targets), model.run_inference(loader_test)))
