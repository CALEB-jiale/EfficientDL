from torchvision.datasets import CIFAR100
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import models_cifar100

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

## Normalization adapted for CIFAR100
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR100 will be downloaded in the following folder
rootdir = '/homes/x21ye/lab1/data/cifar100'

c100train = CIFAR100(rootdir,train=True,download=True,transform=transform_train)
c100test = CIFAR100(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c100train,batch_size=32,shuffle=True)
testloader = DataLoader(c100test,batch_size=32)




## number of target samples for the final dataset
num_train_examples = len(c100train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c100train_subset = torch.utils.data.Subset(c100train,indices[:num_samples_subset])
print(f"Initial CIFAR100 dataset has {len(c100train)} samples")
print(f"Subset of CIFAR100 dataset has {len(c100train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c100train_subset,batch_size=32,shuffle=True)
### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.




# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

backbone_weights_path = '/homes/x21ye/lab1/pretrained_models_cifar100/ResNet18_model_cifar100_lr_0.01.pth'

net = models_cifar100.ResNet18()

if torch.cuda.is_available():
    state_dict=torch.load(backbone_weights_path)
else:
    state_dict=torch.load(backbone_weights_path,map_location=torch.device('cpu'))

net.load_state_dict(state_dict['net'])



# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01)

# # Train the network
# n_epochs=2
# training_losses = []
# for epoch in range(n_epochs):  # loop over the dataset multiple times
#     training_losses.append([])
#     running_loss = 0.0
#     for i, data in enumerate(trainloader_subset, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 20 == 19:    # print every 2000 mini-batches
#             running_loss /= 20
#             training_losses[epoch].append(running_loss)

#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss))
#             running_loss = 0.0

# print('Finished Training')

# PATH_Loss = '/homes/x21ye/lab1/data_log/vgg_cifar100_net_training_loss.npy'
# np.save(PATH_Loss, training_losses)

# PATH = '/homes/x21ye/lab1/nets/vgg_cifar100_net.pth'
# torch.save(net.state_dict(), PATH)

# net.load_state_dict(torch.load(PATH))

accuracies = []

# Test the network on the test data
correct = 0
total = 0
with torch.no_grad():  # torch.no_grad for TESTING
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracies.append(100 * correct / total)
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(100))
class_total = list(0. for i in range(100))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(100):
    accuracies.append(100 * class_correct[i] / class_total[i])
    print('Accuracy of class %5s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))


PATH_Accuracies = '/homes/x21ye/lab1/data_log/resnet_cifar100_net_accuracies.npy'
np.save(PATH_Accuracies, accuracies)