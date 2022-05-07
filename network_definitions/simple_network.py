import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

layers = [(3,5,16),
          (3,16,32),
          (3,32,64),
          (3,64,128),
          (3,128,64),
          (3,64,32),
          (3,32,16),
          (3,16,2)]

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = Conv(8,16)
        self.conv2 = Conv(16,32)
        self.conv3 = Conv(32,64)
        self.conv4 = Conv(64,32)
        self.conv5 = Conv(32,16)
        self.conv6 = Conv(16,1)
        self.sigmoid = Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        logits = self.sigmoid(x)
        return logits
        

"""class SimpleNet(nn.Module):
    def __init__(self, layers, activation="", threshold=False):
        super(SimpleNet, self).__init__()
        self.activation = activation
        cur_layers = []
        for n,(kernel_size,in_channels,out_channels) in enumerate(layers):

            cur_layers.append(Conv2D(kernel_size,in_channels,out_channels))     
            #Reasoning for the use of sigmoid
            #https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
            
            if n != len(layers)-1:    
                #Reasoning of the positioning of Batch Normalization:
                #https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/    
                cur_layers.append(BatchNorm(out_channels))

                if self.activation == "relu":
                    cur_layers.append(nn.ReLU())
                elif self.activation == "sigmoid":
                    cur_layers.append(nn.Sigmoid())
                elif self.activation == "softmax":
                    cur_layers.append(nn.Softmax2d())
                elif self.activation == "tanh":
                    cur_layers.append(nn.Tanh())
                elif self.activation == "lrelu":
                    cur_layers.append(nn.LeakyReLU())
                
        cur_layers.append(Softmax2d())
            
        self.sequential = nn.Sequential(*cur_layers)
        
    def forward(self, x):
        x = self.sequential(x)
        return x

class Conv2D(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(math.floor(kernel_size/2)))
        #print(self.conv2d)
        
    def forward(self,x):
        x = self.conv2d(x)
        return x
    
class BatchNorm(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(features)
        
    def forward(self,x):
        x = self.batchnorm(x)
        return x
    
class Softmax2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.Softmax2d = nn.Softmax2d()
        
    def forward(self,x):
        x = self.Softmax2d(x)
        return x"""

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)