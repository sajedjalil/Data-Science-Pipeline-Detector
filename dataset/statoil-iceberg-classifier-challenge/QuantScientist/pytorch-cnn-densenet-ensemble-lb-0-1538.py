# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from __future__ import print_function
from __future__ import division

# Shlomo Kashani 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import torch
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)

import numpy
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

import pandas
import pandas as pd

import logging
handler=logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)


# !pip install psutil
import psutil
import os
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

cpuStats()

# use_cuda=False
lgr.info("USE CUDA=" + str (use_cuda))


# #  Global params

# In[ ]:


# fix seed
seed=17*19
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)


# #  View the Data
# - Numerai provides a data set that is allready split into train, validation and test sets. 

# In[12]:


# Data params
TARGET_VAR= 'target'
BASE_FOLDER = '../input/'


# #  Train / Validation / Test Split

# In[ ]:


# data = pd.read_json(BASE_FOLDER + '/train.json')
data = pd.read_json("../input/train.json")

print (data.shape)
# data['precision_4'] = data['inc_angle'].apply(lambda x: len(str(x))) <= 7
# data = data[data['precision_4'] == True]
# print (data.shape)


import numpy as np
import scipy.signal

def cross_image(im1, im2):
   # get rid of the color channels by performing a grayscale transform
   # the type cast into 'float' is to avoid overflows
   im1_gray = np.sum(im1.astype('float'), axis=2)
   im2_gray = np.sum(im2.astype('float'), axis=2)

   # get rid of the averages, otherwise the results are not good
   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   # calculate the correlation image; note the flipping of onw of the images
   return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')
   
# Suffle
import random
from datetime import datetime
from scipy import signal
random.seed(datetime.now())
# np.random.seed(datetime.now())
from sklearn.utils import shuffle
data = shuffle(data) # otherwise same validation set each time!
data= data.reindex(np.random.permutation(data.index))

data = shuffle(data) # otherwise same validation set each time!
data= data.reindex(np.random.permutation(data.index))

def Zpad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr

# data['band_1'] = data['band_1'].apply(lambda x: Zpad(x,6400))
# data['band_2'] = data['band_1'].apply(lambda x: Zpad(x,6400))

data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

import scipy
band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
# band_3=(band_1+band_2)/2
# band_3=signal.fftconvolve(band_1, band_1, mode = 'same')

full_img = np.stack([band_1, band_2], axis=1)

# https://github.com/bermanmaxim/jaccardSegment/blob/master/compose.py

# #  From Numpy to PyTorch GPU tensors

# In[14]:


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def XnumpyToTensor(x_data_np):
    x_data_np = np.array(x_data_np, dtype=np.float32)        
    print(x_data_np.shape)
    print(type(x_data_np))

    if use_cuda:
        lgr.info ("Using the GPU")    
        X_tensor = (torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    
    else:
        lgr.info ("Using the CPU")
        X_tensor = (torch.from_numpy(x_data_np)) # Note the conversion for pytorch
        
    print((X_tensor.shape)) # torch.Size([108405, 29])
    return X_tensor


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def YnumpyToTensor(y_data_np):    
    y_data_np=y_data_np.reshape((y_data_np.shape[0],1)) # Must be reshaped for PyTorch!
    print(y_data_np.shape)
    print(type(y_data_np))

    if use_cuda:
        lgr.info ("Using the GPU")            
    #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float        
    else:
        lgr.info ("Using the CPU")        
    #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #         
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float        

    print(type(Y_tensor)) # should be 'torch.cuda.FloatTensor'
    print(y_data_np.shape)
    print(type(y_data_np))    
    return Y_tensor


# #  Custom data loader

# In[17]:

# transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])
# preprocess = transforms.Compose([
#   transforms.Scale(75),
#   transforms.CenterCrop(224),
#   transforms.ToTensor(),
#   normalize
# ])

class FullTrainningDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(FullTrainningDataset, self).__init__()
        
    def __len__(self):        
        return self.length
    
    def __getitem__(self, i):
        # label = torch.from_numpy(self.y_train[index])
        return self.full_ds[i+self.offset]
    
validationRatio=0.11    

def trainTestSplit(dataset, val_share=validationRatio):
    val_offset = int(len(dataset)*(1-val_share))
    print ("Offest:" + str(val_offset))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, 
                                                                              val_offset, len(dataset)-val_offset)


# In[25]:


batch_size=64

from torch.utils.data import TensorDataset, DataLoader

# train_imgs = torch.from_numpy(full_img_tr).float()
train_imgs=XnumpyToTensor (full_img)
train_targets = YnumpyToTensor(data['is_iceberg'].values)
dset_train = TensorDataset(train_imgs, train_targets)


train_ds, val_ds = trainTestSplit(dset_train)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                            num_workers=1)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)

print (train_loader)
print (val_loader)

num_epoches = 5
import math



import attr
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F


n_channels = 2  # max 20
total_classes = 1
    

# https://github.com/Lextal/pspnet-pytorch/blob/master/train.py

import torch
import torch.nn as nn
import torch.nn.functional as Funct

from collections import OrderedDict


# class SegNet(nn.Module):
#     def __init__(self):
#         super(SegNet, self).__init__()

#         self.encoder_1 = nn.Sequential(
#             nn.Conv2d(2, 64, 7, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
#         )  # first group

#         self.encoder_2 = nn.Sequential(
#             nn.Conv2d(64, 64, 7, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
#         )  # second group

#         self.encoder_3 = nn.Sequential(
#             nn.Conv2d(64, 64, 7, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
#         )  # third group

#         self.encoder_4 = nn.Sequential(
#             nn.Conv2d(64, 64, 7, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
#         )  # fourth group

#         self.unpool_1 = nn.MaxUnpool2d(2, stride=2)  # get masks
#         self.unpool_2 = nn.MaxUnpool2d(2, stride=2)
#         self.unpool_3 = nn.MaxUnpool2d(2, stride=2)
#         self.unpool_4 = nn.MaxUnpool2d(2, stride=2)

#         self.decoder_1 = nn.Sequential(
#             nn.Conv2d(64, 64, 7, padding=3),
#             nn.BatchNorm2d(64)
#         )  # first group

#         self.decoder_2 = nn.Sequential(
#             nn.Conv2d(64, 64, 7, padding=3),
#             nn.BatchNorm2d(64)
#         )  # second group

#         self.decoder_3 = nn.Sequential(
#             nn.Conv2d(64, 64, 7, padding=3),
#             nn.BatchNorm2d(64)
#         )  # third group

#         self.decoder_4 = nn.Sequential(
#             nn.Conv2d(64, 3, 7, padding=3),
#             nn.BatchNorm2d(3)
#         )  # fourth group

#         # self.conv_classifier = nn.Conv2d(128, 5, 1)
        
#         self.classifier = torch.nn.Sequential(
#             nn.Linear(972, 1),             
#         )
        
#         self.mp = nn.MaxPool2d(4, 4)
        
#         self.sig = nn.Sigmoid()   

#     def forward(self, x):
#         size_1 = x.size()
#         x, indices_1 = self.encoder_1(x)

#         size_2 = x.size()
#         x, indices_2 = self.encoder_2(x)

#         size_3 = x.size()
#         x, indices_3 = self.encoder_3(x)

#         size_4 = x.size()
#         x, indices_4 = self.encoder_4(x)

#         x = self.unpool_1(x, indices_4, output_size=size_4)
#         x = self.decoder_1(x)

#         x = self.unpool_2(x, indices_3, output_size=size_3)
#         x = self.decoder_2(x)

#         x = self.unpool_3(x, indices_2, output_size=size_2)
#         x = self.decoder_3(x)

#         x = self.unpool_4(x, indices_1, output_size=size_1)
#         x = self.decoder_4(x)
        
#         x = self.mp(x)
#         x = x.view(x.size(0), -1)    
#         print("shape:" + str(x.data.shape))
#         x = self.classifier(x)
#         # print("shape:" + str(x.data.shape))

#         x = self.sig(x)

#         return x


# model=SegNet()

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(2, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(128, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        # print(out.data.shape)
        out = F.avg_pool2d(F.relu(self.bn1(out)), 8)
        out = out.view(out.size(0), -1)
        # print(out.data.shape)
        out = F.sigmoid(self.fc(out))
        return out

model = DenseNet(growthRate=8, depth=20, reduction=0.5,
                            bottleneck=True, nClasses=1)

print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        
print(model)
# https://github.com/ZijunDeng/pytorch-semantic-segmentation/tree/master/models
# https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
# https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py

# # Loss and optimizer

# In[28]:

'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        self.mp = torch.nn.MaxPool2d(1, 1)
        # self.avgpool = torch.nn.AvgPool2d(2,2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))

        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.mp(out)
        # out = self.avgpool(out)

        # print (x.data.shape)

        out = torch.cat([out, x], 1)
        out = self.mp(out)
        # out = self.avgpool(out)
        # print(out.data.shape)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=1):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)

        self.linear = nn.Linear(3328, num_classes)
        self.sig = nn.Sigmoid()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        # print (out.data.shape)
        out = self.linear(out)
        out = self.sig(out)

        return out


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=16)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def densenet_cifar():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)

# test_densenet()



loss_func=torch.nn.BCELoss() # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

# NN params
LR = 0.0005
MOMENTUM= 0.95
optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=5e-5) #  L2 regularization
if use_cuda:
    lgr.info ("Using the GPU")    
    model.cuda()
    loss_func.cuda()

lgr.info (optimizer)
lgr.info (loss_func)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

criterion = loss_func
all_losses = []
val_losses = []


if __name__ == '__main__':

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch + 1, num_epoches))
        print('*' * 5 + ':')
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
    
            img, label = data
            if use_cuda:
                img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))  # On GPU
            else:
                img, label = Variable(img), Variable(
                    label)  # RuntimeError: expected CPU tensor (got CUDA tensor)
    
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.data[0] * label.size(0)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if i % 10 == 0:
                all_losses.append(running_loss / (batch_size * i))
                print('[{}/{}] Loss: {:.6f}'.format(
                    epoch + 1, num_epoches, running_loss / (batch_size * i),
                    running_acc / (batch_size * i)))
    
        print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, running_loss / (len(train_ds))))
    
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in val_loader:
            img, label = data
    
            if use_cuda:
                img, label = Variable(img.cuda(async=True), volatile=True),
                Variable(label.cuda(async=True), volatile=True)  # On GPU
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
    
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)
    
        print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_ds))))
        val_losses.append(eval_loss / (len(val_ds)))
        print()
    
    torch.save(model.state_dict(), './cnn.pth')
    
    
    df_test_set = pd.read_json('../input/test.json')
    
    df_test_set['band_1'] = df_test_set['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    df_test_set['band_2'] = df_test_set['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    df_test_set['inc_angle'] = pd.to_numeric(df_test_set['inc_angle'], errors='coerce')
    
    df_test_set.head(3)
    
    
    print (df_test_set.shape)
    columns = ['id', 'is_iceberg']
    df_pred=pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
    # df_pred.id.astype(int)
    
    for index, row in df_test_set.iterrows():
        rwo_no_id=row.drop('id')    
        band_1_test = (rwo_no_id['band_1']).reshape(-1, 75, 75)
        band_2_test = (rwo_no_id['band_2']).reshape(-1, 75, 75)
        full_img_test = np.stack([band_1_test, band_2_test], axis=1)
    
        x_data_np = np.array(full_img_test, dtype=np.float32)        
        if use_cuda:
            X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    
        else:
            X_tensor_test = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch
                        
    #     X_tensor_test=X_tensor_test.view(1, trainX.shape[1]) # does not work with 1d tensors            
        predicted_val = (model(X_tensor_test).data).float() # probabilities     
        p_test =   predicted_val.cpu().numpy().item() # otherwise we get an array, we need a single float
        
        df_pred = df_pred.append({'id':row['id'], 'is_iceberg':p_test},ignore_index=True)
    #     df_pred = df_pred.append({'id':row['id'].astype(int), 'probability':p_test},ignore_index=True)
    
    df_pred.head(5)
    
    
    def savePred(df_pred):
    #     csv_path = 'pred/p_{}_{}_{}.csv'.format(loss, name, (str(time.time())))
    #     csv_path = 'pred_{}_{}.csv'.format(loss, (str(time.time())))
        csv_path='sample_submission.csv'
        df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
        print (csv_path)
        
    savePred (df_pred)