# PyTorch CNN CIFAR10 Training Script

# Example translated from keras model desribed in:
#   https://www.tensorflow.org/tutorials/images/cnn (as of November 19, 2020)

import torch
import torchvision as tv
import cnn_model

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('pytorch-profile-train')

transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])

# CIFAR10 labels: ['plane', 'car', 'bird', 'cat', 'deer',
#                  'dog', 'frog', 'horse', 'ship', 'truck']
train = tv.datasets.CIFAR10(root='./cifar10-data', transform=transform,
                            download=True, train=True)
test  = tv.datasets.CIFAR10(root='./cifar10-data', transform=transform,
                            download=True, train=False)
print('training-set: %d images' % len(train.targets))
print('test-set:     %d images' % len(test.targets))

batch_size = 32
loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                           num_workers=2, shuffle=True)
loader_test  = torch.utils.data.DataLoader(test,  batch_size=batch_size,
                                           num_workers=2, shuffle=False)

model = cnn_model.CnnModel()

cross_entropy = torch.nn.CrossEntropyLoss()            # Loss function
adam = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer

for epoch in range(2):

  for step, data in enumerate(loader_train, 0):
    inputs, labels = data
    adam.zero_grad()
    outputs = model(inputs)
    batch_loss = cross_entropy(outputs, labels)
    batch_loss.backward()
    adam.step()

    if step == 1:
        save_images = inputs

    if step % 100 == 99:

        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader_test:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(('Epoch: %d, Step %5d, Images: %5d  loss: ' +
              '%3f  test-set accuracy: %5.2f') %
              (epoch + 1, step + 1, (step + 1) * batch_size,
               batch_loss.item(), model.run_inference(loader_test)))
        accumulated_loss = 0.0

torch.save(model.state_dict(), model.save_filename())

writer.add_graph(model, save_images)
writer.close()
