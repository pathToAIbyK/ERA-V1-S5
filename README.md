# ERA-V1-S5 Assignment

This model works on the MNIST dataset to detect handwritten digits.

## Import the libraries
```diff
import torch #Machine learning library having Tensors, Neural Networks, Optimization Algorithms and also Visualization tools.
import torch.nn as nn #contains classes and functions which are used to create neural networks.
import torch.nn.functional as F #This module provides fuctions such as activation, pooling, loss, padding etc.
import torch.optim as optim #this module helps in training the NN by a collection of optimization algorithms.
from torchvision import datasets, transforms #we are importing CV libraries like datasets and transforms that prepare data for CV tasks.
!pip install torchinfo
from torchinfo import summary #this will help us in understanding the models structure, parameters and memory usage<br> 
```
## Swithch the device to GPU
```diff
use_cuda = torch.cuda.is_available() #it is a boolean fuction which returns true/false depending if CUDA enabled GPU is enabled.
device = torch.device("cuda" if use_cuda else "cpu") #assigns the device to GPU
device
```
## Clone the Git repositry
### Model.py and Utils.py
```diff
!git clone https://github.com/pathToAIbyK/ERA-V1-S5.git
%cd ERA-V1-S5
```
## Run the model file to check parameters
```diff
!python model.py
import model as model
model = model.Net().to(device)
summary(model, input_size=(1, 28, 28))
```
```diff
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      [1, 10]                   --
├─Conv2d: 1-1                            [32, 26, 26]              320
├─Conv2d: 1-2                            [64, 24, 24]              18,496
├─Conv2d: 1-3                            [128, 10, 10]             73,856
├─Conv2d: 1-4                            [256, 8, 8]               295,168
├─Linear: 1-5                            [1, 50]                   204,850
├─Linear: 1-6                            [1, 10]                   510
==========================================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
Total mult-adds (M): 727.92
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 2.37
Estimated Total Size (MB): 3.08
==========================================================================================
```
## Run the Utils file
This file has drives the following actions :
- Create a Train and Test set
- Load the test loader and train loader
- Add mutliple tranforms for the train set and keep the test set as simple as possible
- Create a Train and test functions which backpropagate and calculated loss
```diff
!python utils.py
import utils as utils
```
## Use the SGD Optimizer with the learning rate of 0.01
```
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```
## Scheduler assigned with respect to the optimizer
```
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
```
## Add the criterion to calculate the cross entropy loss between input and target
```
criterion = nn.CrossEntropyLoss()
```
## Run the network to train and test.
```
num_epochs = 2

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  utils.train(model, device, utils.train_loader, optimizer, criterion)
  scheduler.step()
  utils.test(model, device, utils.test_loader,criterion)
```
