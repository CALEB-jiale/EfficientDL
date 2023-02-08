import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

import sys
sys.path.append("pytorch-cifar-master")
from models import *

PATH = '/Users/lucas/Documents/School/IMT/A2/S4/EfficientDL/02_01/lab1'

# Load data
models = ["VGG16", "ResNet18", "ResNet50", "ResNet101", "RegNetX_200MF", "RegNetY_400MF", "MobileNetV2",
          "ResNeXt29(32x4d)", "ResNeXt29(2x64d)", "SimpleDLA", "DenseNet121", "PreActResNet18", "DPN92", "DLA"]
acc = [92.64, 93.02, 93.62, 93.75, 94.24, 94.29, 94.43,
       94.73, 94.82, 94.89, 95.04, 95.11, 95.16, 95.47]


# Get number of parameters
num_parameters = []

net = VGG("VGG16")
num_parameters.append(summary(net).total_params)
net = ResNet18()
num_parameters.append(summary(net).total_params)
net = ResNet50()
num_parameters.append(summary(net).total_params)
net = ResNet101()
num_parameters.append(summary(net).total_params)
net = RegNetX_200MF()
num_parameters.append(summary(net).total_params)
net = RegNetY_400MF()
num_parameters.append(summary(net).total_params)
net = MobileNetV2()
num_parameters.append(summary(net).total_params)
net = ResNeXt29_32x4d()
num_parameters.append(summary(net).total_params)
net = ResNeXt29_2x64d()
num_parameters.append(summary(net).total_params)
net = SimpleDLA()
num_parameters.append(summary(net).total_params)
net = DenseNet121()
num_parameters.append(summary(net).total_params)
net = PreActResNet18()
num_parameters.append(summary(net).total_params)
net = DPN92()
num_parameters.append(summary(net).total_params)
net = DLA()
num_parameters.append(summary(net).total_params)

# Plotting the graph
plt.plot(num_parameters, acc, 'ro')
plt.xlabel('Number of Parameters (M)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Number of Parameters')
for i, model in enumerate(models):
    plt.annotate(model, (num_parameters[i], acc[i]))

plt.savefig(PATH + "/imgs/AccVSNumPara.png")
plt.show()