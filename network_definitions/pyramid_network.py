import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

class PyramidNet(nn.Module):
    def __init__(self, n_channels, inner_layers, activation="", threshold=False):
        super(PyramidNet, self).__init__()
        self.n_channels = n_channels
        self.activation = activation
        self.branches = []
        
        if self.activation == "relu":
            self.activation_layer = nn.ReLU()
        elif self.activation == "sigmoid":
            self.activation_layer = nn.Sigmoid()
        elif self.activation == "softmax":
            self.activation_layer = nn.Softmax2d()
        elif self.activation == "tanh":
            self.activation_layer = nn.Tanh()
        
        self.last = nn.Sequential(Conv2D(15,1,True),
                                  BatchNorm(True),
                                  self.activation_layer)
        
        layers = [1,3,5]
        
        for n in layers:
            cur_layers = []
            
            for k in range(0,inner_layers):
                cur_layers.append(Conv2D(5,n,False))

            
            self.branches.append(nn.Sequential(*cur_layers))

        if threshold:
            self.threshold = nn.Threshold(0.5)

    def forward(self, x):
        b1 = self.branches[0](x)
        b2 = self.branches[1](x)
        b3 = self.branches[2](x)
        c1 = torch.concat([b1,b2,b3])
        x = self.last(c1)
        return x

class Conv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, last):
        super().__init__()
        if last:
            out_channels = 1
        else:
            out_channels = 5
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(math.floor(kernel_size/2)))
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