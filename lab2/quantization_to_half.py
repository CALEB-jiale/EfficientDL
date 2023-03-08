from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
import warnings
import sys
sys.path.append("pytorch-cifar-master")
from models import *
from utils import progress_bar

warnings.filterwarnings("ignore")

PATH = '/homes/x21ye/Documents/Efficient DL'
PATH_net = PATH + '/lab2/nets/dense_net_cifar10_0001.pth'
# PATH_net = PATH + '/lab1/nets/VGG16_cifar_net_0001.pth'

#  The data from CIFAR10 will be downloaded in the following folder
rootdir = PATH + '/lab1/data_set/cifar10'

#  Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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

c10train = CIFAR10(rootdir, train=True, download=True,
                   transform=transform_train)
c10test = CIFAR10(rootdir, train=False, download=True,
                  transform=transform_test)

trainloader = DataLoader(c10train, batch_size=32, shuffle=True)
testloader = DataLoader(c10test, batch_size=32)


# number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

# We set a seed manually so as to reproduce the results easily
seed = 2147483647

# Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(
    indices)  #  modifies the list in place

# We define the Subset using the generated indices
c10train_subset = torch.utils.data.Subset(
    c10train, indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset, batch_size=32, shuffle=True)
# You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.


# Train the network
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.half().to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        tot_time = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('Finished Training')
    train_acc = 100.*correct/total
    return train_acc, train_loss/(batch_idx+1), tot_time


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.half().to(device), targets.to(device)
            # inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tot_time = progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total

    return acc, test_loss/(batch_idx+1), tot_time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = DenseNet121()
# net = VGG('VGG16')
net.load_state_dict(torch.load(PATH_net))
net.half()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

test_acc, test_loss, test_time = test(0)

# train_loss = []
# train_acc = []
# train_time = []
# test_loss = []
# test_acc = []
# test_time = []
# for epoch in range(0, 200):
#     test_acc_epoch, test_loss_epoch, test_time_epoch = test(epoch)
#     scheduler.step()
#     train_loss.append(train_loss_epoch)
#     train_acc.append(train_acc_epoch)
#     train_time.append(train_time_epoch)
#     test_loss.append(test_loss_epoch)
#     test_acc.append(test_acc_epoch)
#     test_time.append(test_time_epoch)

# PATH_net = PATH + '/lab2/nets/dense_net_half_cifar10_0001.pth'
# torch.save(net.state_dict(), PATH_net)

# PATH_data = PATH + '/lab2/data_log/dense_net_half_cifar10_0001_'

# PATH_train_acc = PATH_data + 'train_acc.npy'
# np.save(PATH_train_acc, train_acc)

# PATH_train_loss = PATH_data + 'train_loss.npy'
# np.save(PATH_train_loss, train_loss)

# PATH_train_time = PATH_data + 'train_time.npy'
# np.save(PATH_train_time, train_time)

# PATH_test_acc = PATH_data + 'test_acc.npy'
# np.save(PATH_test_acc, test_acc)

# PATH_test_loss = PATH_data + 'test_loss.npy'
# np.save(PATH_test_loss, test_loss)

# PATH_test_time = PATH_data + 'test_time.npy'
# np.save(PATH_test_time, test_time)