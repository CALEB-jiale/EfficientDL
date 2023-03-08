from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from matplotlib import pyplot as plt 

PATH = '/homes/x21ye/Documents/Efficient DL'

#  The data from CIFAR10 will be downloaded in the following folder
rootdir = PATH + '/lab1/data_set/cifar10'

#  Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network.
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
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

trainloader = DataLoader(c10train, batch_size=4, shuffle=False)


path_imgs = PATH + "/lab4/imgs"

### Let's do a figure for each batch
f = plt.figure(figsize=(10,10))

for i,(data,target) in enumerate(trainloader):
    
    data = (data.numpy())
    print(data.shape)
    plt.subplot(2,2,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))

    break

f.savefig(path_imgs + '/train_DA_RandomPerspective.png')


































# testloader = DataLoader(c10test, batch_size=32)


# # number of target samples for the final dataset
# num_train_examples = len(c10train)
# num_samples_subset = 15000

# # We set a seed manually so as to reproduce the results easily
# seed = 2147483647

# # Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
# indices = list(range(num_train_examples))
# np.random.RandomState(seed=seed).shuffle(
#     indices)  #  modifies the list in place

# # We define the Subset using the generated indices
# c10train_subset = torch.utils.data.Subset(
#     c10train, indices[:num_samples_subset])
# print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
# print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# # Finally we can define anoter dataloader for the training data
# trainloader_subset = DataLoader(c10train_subset, batch_size=32, shuffle=True)
# # You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.
