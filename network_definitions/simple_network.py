import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

class SimpleNet(nn.Module):
    def __init__(self, n_channels, layers, activation="", threshold=False):
        super(SimpleNet, self).__init__()
        self.n_channels = n_channels
        self.activation = activation
        cur_layers = []
        for n,kernel_size in enumerate(layers):
            if n == len(layers)-1:
                last = True
            else:
                last = False
            cur_layers.append(Conv2D(kernel_size,last))
            
            #Reasoning for the use of sigmoid
            #https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
            
                
            #Reasoning of the positioning of Batch Normalization:
            #https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/    
            cur_layers.append(BatchNorm(last))
            
            if self.activation == "relu":
                cur_layers.append(nn.ReLU())
            elif self.activation == "sigmoid":
                cur_layers.append(nn.Sigmoid())
            elif self.activation == "softmax":
                cur_layers.append(nn.Softmax2d())
            elif self.activation == "tanh":
                cur_layers.append(nn.Tanh())
            
        self.sequential = nn.Sequential(*cur_layers)

        self.class_branch = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, stride=2),
                                          nn.Conv2d(1, 1, kernel_size=3, stride=2),
                                          nn.Conv2d(1, 1, kernel_size=3, stride=2),
                                          nn.Flatten(),
                                          nn.LazyLinear(1024),
                                          nn.LeakyReLU(),
                                          nn.LazyLinear(1),
                                          nn.Sigmoid())
        
        self.threshold = threshold

    def forward(self, x):
        x = self.sequential(x)
        classification = self.class_branch(x)
        return (x,classification)

class Conv2D(nn.Module):
    def __init__(self, kernel_size, last):
        super().__init__()
        if last:
            out_channels = 1
        else:
            out_channels = 5
        self.conv2d = nn.Conv2d(5, out_channels, kernel_size=kernel_size, padding=int(math.floor(kernel_size/2)))
        #print(self.conv2d)
        
    def forward(self,x):
        x = self.conv2d(x)
        return x
    
class BatchNorm(nn.Module):
    def __init__(self, last):
        super().__init__()
        if last:
            features = 1
        else:
            features = 5
        self.batchnorm = nn.BatchNorm2d(features)
        
    def forward(self,x):
        x = self.batchnorm(x)
        return x