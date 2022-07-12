# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import genfromtxt
import pandas as pd

from torch.utils.data.dataset import Dataset
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''
dir_filenames=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path=os.path.join(dirname, filename)
        print(path)
        dir_filenames.append(path)
# Any results you write to the current directory are saved as output.
'''




    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(4,self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(4608, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x=x.permute(0,3,1,2)
        #print(x.type())
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        #print(out.shape)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

class dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,  root_dir):
        
        #root_dir='E:\\IPSL\\Remote_Sensing\\Graduation Reserach\\Data\\High Res\\Cropped\\Train'
        self.root_dir=root_dir
        #data = genfromtxt(self.root_dir+'train_labels.csv', delimiter=',', names=True)
        self.label=pd.read_csv(self.root_dir+'train_labels.csv') 
        
        #self.id=pd.read_csv(self.root_dir+'train_labels.csv', usecols =["id"]) 
        #print(self.id.shape)
        self.TotalImg = self.label.shape[0]
      

    def __len__(self):
        return self.TotalImg+1

    def __getitem__(self, idx):
        label=self.label['label'][idx]
        path_img=self.label['id'][idx]
        #print(label)
        img=Image.open(self.root_dir+'train/'+path_img+'.tif')
        #print(img.shape)
        img=np.array(img).astype(float)/255
        
        img=torch.from_numpy(img)
        
        #label=torch.from_numpy(label)
        #print('cek')
        #return timg.type(torch.cuda.FloatTensor),tGT.type(torch.cuda.FloatTensor),tant.type(torch.cuda.FloatTensor)
        return  img,label
class Solver(object):
    def __init__(self,train_loader):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader=train_loader
        self.criterion=nn.CrossEntropyLoss()
        self.learning_rate = 0.01
        self.model=ResNet18().to(self.device)
        self.optimizer=optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
       
    def train(self):
        for epoch in range(10):
            print('\nEpoch: %d' % epoch)
            self.model.train(True)
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx,(images, targets) in enumerate(self.train_loader):
                #print(batch_idx)
                images = images.to(self.device).type(torch.FloatTensor)
                targets = targets.to(self.device).type(torch.long)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                #print(targets)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc=100.*correct/total
               
                if (batch_idx+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, acc:{:.3f}% -- ({}/{}) ' 
                       .format(epoch+1, 100, batch_idx+1, len(self.train_loader),train_loss/(batch_idx+1),acc, correct, total))
                
                
              

if __name__ == '__main__':
    path_train='/kaggle/input/histopathologic-cancer-detection/'
    path_test='/kaggle/input/histopathologic-cancer-detection/test/'
    Dataset_train=dataset(path_train)
    train_loader = torch.utils.data.DataLoader(dataset=Dataset_train,
                                                    batch_size=4,
                                                    shuffle=True)
    
    solver=Solver(train_loader)
    solver.train()
    
    
    
    
    
    
    