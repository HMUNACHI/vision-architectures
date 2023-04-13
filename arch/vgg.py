import torch
import torch.nn as nn

class VGG(nn.Module):
  '''
  Implementation of VGG with Batch Norm for MNIST and CIFAR
  num_classes (int): number of existing classes in data
  '''
  def __init__(self, arch='VGG13', channels = 1, num_classes = 10):
    super(VGG, self).__init__()

    self.config = {
        'VGG13': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
        'VGG16': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool'],
        'VGG19': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 256, 'pool', 512, 512, 512, 512, 'pool', 512, 512, 512, 512, 'pool']
        }
    self.in_planes = channels
    self.convs = self.stack_layers(self.config[arch])

    self.fully_connected = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace = True),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(inplace = True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

  def stack_layers(self, architecture):
    in_planes = self.in_planes
    stack = []

    for layer in architecture:
      if type(layer) == int:
        out = layer
        stack += [nn.Conv2d(in_planes, out, kernel_size = 3, stride = 1, padding = 1),
                  nn.BatchNorm2d(layer),
                  nn.ReLU(inplace = True)]
        in_planes = layer
      else:
        stack += [nn.MaxPool2d(kernel_size = 2)]
      
    return nn.Sequential(*stack)

  def forward(self, x):
    x = self.convs(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fully_connected(x)

    return x