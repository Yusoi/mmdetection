BATCH_SIZE = 1
N_CHANNELS = 4

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from network_definitions.u_net import UNet
from network_definitions.fcn import FCN32s as FCN
from network_definitions.simple_network import SimpleNet
from network_definitions.pyramid_network import PyramidNet
from torchvision.models.segmentation import fcn_resnet101 as FCN_Res101

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class PatchesDatasetTrain(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inp = io.imread(self.root_dir+"/inp/"+str(idx)+".png")
        gt = io.imread(self.root_dir+"/gt/"+str(idx)+".png",as_gray=True)
        
        sample = {'name': idx, 'inp': inp, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class PatchesDatasetTest(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 4629

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inp = io.imread(self.root_dir+"/inp/"+str(idx+20000)+".png")
        gt = io.imread(self.root_dir+"/gt/"+str(idx+20000)+".png",as_gray=True)
        
        sample = {'name': idx, 'inp': inp, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
    
from skimage.transform import resize
from torchvision import transforms, utils
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        name,inp,gt = sample["name"],sample["inp"],sample["gt"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        inp = inp.transpose((2, 0, 1))
        gt = gt[:,:,np.newaxis].transpose((2, 0, 1))
        return {"name": name, 
                "inp": torch.from_numpy(inp),
                "gt": torch.from_numpy(gt)}
        
trainset = PatchesDatasetTrain("data/cityscapes_patches", 
                               transform=transforms.Compose([ToTensor()]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=6)

from torch.utils.tensorboard import SummaryWriter

#PATH = "work_dirs/simplenet_1/"


import torch.optim as optim

def train(net, trainloader, criterion, optimizer, save_path, tensorboard_path, checkpoint=None):
    
    EPOCH = 0
    
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    if checkpoint != None:
        checkpoint = torch.load(checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        EPOCH = checkpoint['epoch']
        loss = checkpoint['loss']
        net.train()
    
    for epoch in range(EPOCH,50):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            im_seg = data["inp"].to(device, dtype=torch.float)
            im_res = data["gt"].to(device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            output = net(im_seg.float())
            
            loss = criterion(output.float(), im_res.float())
            loss.backward(retain_graph=True)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                """print('[%d, %5d] segm loss: %.6f  class loss: %.6f  loss: %.6f' %
                      (epoch + 1, i + 1, running_loss_segm / 50, running_loss_class / 50, running_loss / 50))"""
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 1999))
                running_loss = 0.0
                
                input_ = im_seg.cpu().detach()
                output_ = output.cpu().detach()
                output_ = torch.argmax(output_,1)
                #print(output_.shape)
                gt_output_ = im_res.cpu().detach()
                
                input_ = input_.numpy()[0].transpose((1,2,0))
                output_ = output_.numpy().transpose((1,2,0))
                
                gt_output_ = gt_output_.numpy()[0].transpose((1,2,0)).squeeze(axis=2)
                
                fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
                ax=ax.flat
                    
                ax[0].set_title("Original Image Patch")  # set title
                ax[0].imshow(input_[:,:,0:3])
                
                #ax.append(fig.add_subplot(2, 4, 7))
                ax[1].set_title("Boundary Input")  # set title
                ax[1].imshow(input_[:,:,3],cmap='gray')
                
                ax[2].set_title("Boundary Output")
                ax[2].imshow(output_,cmap='gray')
                
                ax[3].set_title("Ground Truth")
                ax[3].imshow(gt_output_,cmap='gray')
                
                fig.tight_layout()
                plt.show()
                
                print("Max Value: ",output_.max()," Min Value: ",output_.min())
            
        writer.add_scalar('Loss', loss, epoch)

        #if epoch % 5 == 4:        
        torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, save_path+"epoch_"+str(epoch+1)+".pt")    
    
    writer.close()

    print('Finished Training')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OPTIMIZER = "SGD"
LOSS = "BCELoss"

print("Starting training on network: UNet ")

net = UNet(4,1)
net = net.to(device).float()

if LOSS == "BCELoss":
    criterion = nn.BCELoss()
elif LOSS == "CrossEntropyLoss":
    criterion = nn.CrossEntropyLoss()

if OPTIMIZER == "SGD":
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
elif OPTIMIZER == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=0.001)

checkpoint_path = "work_dirs/unet_boundary_refinement/"
tensorboard_path = checkpoint_path+"tb/"
os.makedirs(tensorboard_path,exist_ok=True)

train(net,trainloader,criterion,optimizer, checkpoint_path, tensorboard_path)#, checkpoint="work_dirs/simplenet_1/epoch_25.pt")