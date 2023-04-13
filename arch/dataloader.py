import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

def load_data(name, batch_size):
  # number of subprocesses to use for data loading
  num_workers = 0
  # percentage of training set to use as validation
  valid_size = 0.1

  # convert data to a normalized,rotational and translational invariaant torch.FloatTensor
  transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

  # choose the training and test datasets
  if name == 'MNIST':
    train_data = datasets.MNIST('data', train=True,
                              download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False,
                             download=True, transform=transform)
  elif name == 'CIFAR10':
    train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)
  else:
    print('Warning: Dataset unavailable')

  # obtain training indices for validation
  num_train = len(train_data)
  indices = list(range(num_train))
  np.random.shuffle(indices)
  split = int(np.floor(valid_size * num_train))
  train_idx, valid_idx = indices[split:], indices[:split]

  # defining samplers for obtaining training and validation batches
  train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
  valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

  # preparing data loaders
  train_loader = torch.utils.data.DataLoader(
              train_data, 
              batch_size=batch_size,
              sampler=train_sampler, 
              num_workers=num_workers)

  valid_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=batch_size, 
                                           sampler=valid_sampler, 
                                           num_workers=num_workers)

  test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=batch_size, 
                                        num_workers=num_workers)

  # Explore the structure
  print('Train:', len(train_data))
  print('Test:', len(test_data))

  dataiter = iter(train_loader)
  images, labels = dataiter.next()
  images = images.numpy() # convert images to numpy for display
  print('Image structure:', images.shape)
  
  fig = plt.figure(figsize = (25, 4))

  for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    if name == 'MNIST':
      ax.imshow(np.squeeze(images[idx]), cmap = 'gray' )
      ax.set_title(str(labels[idx].item()))

    elif name == 'CIFAR10':
      # specifying the image classes
      classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

      # helper function to un-normalize and display an image
      def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
        
      imshow(images[idx])
      ax.set_title(classes[labels[idx]])

    else:
     print('No Data')
     break

  return train_loader, test_loader, valid_loader