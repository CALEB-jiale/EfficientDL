from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
import torch.optim as optim
import warnings
import argparse
import sys
from mix_up import *
sys.path.append("pytorch-cifar-master")
from models import *
from utils import progress_bar


warnings.filterwarnings("ignore")
PATH = '/homes/x21ye/Documents/Efficient DL'

#  The data from CIFAR10 will be downloaded in the following folder
data_dir = PATH + '/lab1/data_set/cifar10'
data_checkpoint = PATH + '/lab4//checkpoint/ckpt.pth'


def load_data(data_dir):
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

    c10train = CIFAR10(data_dir, train=True, download=True,
                    transform=transform_train)
    c10test = CIFAR10(data_dir, train=False, download=True,
                    transform=transform_test)

    trainloader = DataLoader(c10train, batch_size=32, shuffle=True)
    testloader = DataLoader(c10test, batch_size=32)
    return trainloader, testloader

# Train the network
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    mix = False
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if mix:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)

        optimizer.zero_grad()
        outputs = net(inputs)

        _, predicted = outputs.max(1)
        total += targets.size(0)

        if mix:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        else:
            loss = criterion(outputs, targets)
            correct += predicted.eq(targets).sum().item()

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        tot_time = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if correct/total > 0.5 and not mix:
            print("Start mix up")
            print()
            mix = True

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
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tot_time = progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, data_checkpoint)
        best_acc = acc

    return acc, test_loss/(batch_idx+1), tot_time


# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainloader, testloader = load_data(data_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

net = DenseNet121()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_loss = []
train_acc = []
train_time = []
test_loss = []
test_acc = []
test_time = []

PATH_net = PATH + '/lab4/nets/dense_net_cifar10_0001.pth'
PATH_data = PATH + '/lab4/data_log/dense_net_cifar10_0001_mixed_'
PATH_train_acc = PATH_data + 'train_acc.npy'
PATH_train_loss = PATH_data + 'train_loss.npy'
PATH_train_time = PATH_data + 'train_time.npy'
PATH_test_acc = PATH_data + 'test_acc.npy'
PATH_test_loss = PATH_data + 'test_loss.npy'
PATH_test_time = PATH_data + 'test_time.npy'


for epoch in range(0, 1):
    train_acc_epoch, train_loss_epoch, train_time_epoch = train(epoch)
    test_acc_epoch, test_loss_epoch, test_time_epoch = test(epoch)
    scheduler.step()
    train_loss.append(train_loss_epoch)
    train_acc.append(train_acc_epoch)
    train_time.append(train_time_epoch)
    test_loss.append(test_loss_epoch)
    test_acc.append(test_acc_epoch)
    test_time.append(test_time_epoch)

    
    torch.save(net.state_dict(), PATH_net)
    np.save(PATH_train_acc, train_acc)
    np.save(PATH_train_loss, train_loss)
    np.save(PATH_train_time, train_time)
    np.save(PATH_test_acc, test_acc)
    np.save(PATH_test_loss, test_loss)
    np.save(PATH_test_time, test_time)