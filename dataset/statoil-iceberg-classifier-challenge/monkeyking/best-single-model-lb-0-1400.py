# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: perry
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook
import seaborn as sns

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets, models
import random
import PIL
from PIL import Image, ImageOps
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import time
import copy


def iso(arr):
    p = np.reshape(np.array(arr), [75,75]) >(np.mean(np.array(arr))+2*np.std(np.array(arr)))
    return p * np.reshape(np.array(arr), [75,75])
def size(arr):     
    return float(np.sum(arr<-5))/(75*75)

data = pd.read_json('.../iceberg/train.json')
test = pd.read_json('.../iceberg/test.json')

data['iso1'] = data.iloc[:, 0].apply(iso)
data['iso2'] = data.iloc[:, 1].apply(iso)
test['iso1'] = test.iloc[:, 0].apply(iso)
test['iso2'] = test.iloc[:, 1].apply(iso)
# Feature engineering s1 s2 and size.
data['s1'] = data.iloc[:,5].apply(size)
data['s2'] = data.iloc[:,6].apply(size)
test['s1'] = test['iso1'].apply(size)
test['s2'] = test['iso2'].apply(size)

data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
test['band_1'] = test['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
test['band_2'] = test['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')



#####process test set!
band_1_test = np.concatenate([im for im in test['band_1']]).reshape(-1, 75, 75)
band_2_test = np.concatenate([im for im in test['band_2']]).reshape(-1, 75, 75)
full_img_test = np.stack([band_1_test, band_2_test,(band_1_test+band_2_test)/2], axis=1)
angle_test=test['inc_angle']
size_test=test['s1']
test['is_iceberg']=0


def test_totensor(img):
    img= img.astype(float)/255
    tensor=torch.from_numpy(img.copy())
    return tensor

def do_clip(arr, mx): 
    return np.clip(arr, (1-mx), mx)



#########augmentation
class read_data(Dataset):
    """total dataset."""

    def __init__(self, data, labels, angle,size,transform=None):

        self.data= data
        self.labels = labels
        self.transform = transform
        self.angle = angle
        self.size = size
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx,:,:,:], 'labels': np.asarray([self.labels.values[idx]]), 'angle': np.asarray([self.angle.values[idx]]),'size': np.asarray([self.size.values[idx]])}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels, angle,size = sample['image'], sample['labels'], sample['angle'],sample['size']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        image = image.astype(float)/255
        return {'image': torch.from_numpy(image.copy()).float(),
                'labels': torch.from_numpy(labels).long(),
                'angle': torch.from_numpy(angle).float(),
                'size': torch.from_numpy(size).float()}



class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels, angle,size = sample['image'], sample['labels'], sample['angle'],sample['size']

        
        if random.random() < 0.5:
            image=np.flip(image,1)
        
        return {'image': image, 'labels': labels,'angle':angle,'size':size}

class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels, angle,size = sample['image'], sample['labels'], sample['angle'],sample['size']
        
        if random.random() < 0.5:
            image=np.flip(image,0)
        
        return {'image': image, 'labels': labels, 'angle':angle,'size':size} 

#####for all the training and network function

use_gpu = torch.cuda.is_available()
from tqdm import tqdm
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 10000.0
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True) # Set model to training mode
                dataloader=train_loader
                dataset_sizes=len(train_dataset)
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader=val_loader
                dataset_sizes=len(val_dataset)
                
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloader):
                
                # get the inputs
                inputs, labels, angle,size = data['image'], data['labels'], data['angle'],data['size']
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    angle = Variable(angle.cuda())
                    size = Variable(size.cuda())
                else:
                    inputs, labels, angle = Variable(inputs), Variable(labels), Variable(angle)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs.float(),size.float())
                #print(outputs.float())
                loss = criterion(outputs.float(), labels.resize((len(labels))).long())
                _, preds = torch.max(outputs.data, 1) #for classification
                

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]*len(labels)
                running_corrects += torch.sum(preds == labels.resize((len(labels))).long().data)

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = float(running_corrects) / dataset_sizes

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
               phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f} Best val acc: {:4f}'.format(best_loss,best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,best_loss,best_acc


class Vgg11bnNet(nn.Module):
    def __init__(self, num_classes=2):
        super(Vgg11bnNet, self).__init__()
        self.features = nn.Sequential(*list(model_vgg.features.children()))# 28 is total
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3,1,padding=1),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.6),
            nn.BatchNorm2d(512),
            #nn.MaxPool2d(kernel_size=2,stride=1),#7*7->6*6
            
            nn.Conv2d(512, 128, 3,1,padding=1),
            nn.ReLU(inplace=True),

            
        )
        self.fc = nn.Linear(129,2) #need to be considered
        self.dropout=nn.Dropout()
        self._initialize_weights()
        
    def forward(self, x,size):
        x = self.features(x)
        x = self.classifier(x)
        r = x.size(3)       
        x = F.avg_pool2d(x, r)
        x = x.view(x.size(0), -1)
        x = torch.cat((x,size),1)
        x = self.fc(F.relu(x))
        return x
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()  


def predict(input_model,loader,dataset,phase):
    
    dataloader=loader
    dataset_sizes=len(dataset)
    running_loss=0.0
    running_corrects=0
    test_score=[]
    #gt=[]
    for data in dataloader:
        
        # get the inputs
        inputs, labels, angle,size = data['image'], data['labels'], data['angle'],data['size']
        #gt.extend(labels.numpy())
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            angle = Variable(angle.cuda())
            size = Variable(size.cuda())

        # forward
        outputs = input_model(inputs.float(),size.float())
        if phase=='val':
            #print(outputs.float())
            loss = criterion(outputs.float(), labels.resize((len(labels))).long())
            _, preds = torch.max(outputs.data, 1) #for classification
    
            # statistics
            running_loss += loss.data[0]*len(labels)
            running_corrects += torch.sum(preds == labels.resize((len(labels))).long().data)
        m=nn.Softmax()
        probs=m(outputs)
        probs_value=probs.cpu().data.numpy()
        test_score.extend(probs_value[:,1])
     
    score=np.array(test_score)    
    #softmax+log+nll-loss=crossentropy loss (nll-loss is log loss in neural network)
    if phase=='val':
        epoch_loss = running_loss / dataset_sizes
        epoch_acc = float(running_corrects) / dataset_sizes
    
        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
        return score,epoch_loss
    else:
        return score

n_fold=10
from sklearn.model_selection import KFold

kf = KFold(n_splits=n_fold, shuffle=True)
i = 0
train_error=[]
val_error=[]
#######cross validation start here
for train_ind, val_ind in kf.split(data):
    
    
    train=data.loc[train_ind]
    val=data.loc[val_ind]

    train=train.reset_index()
    del train['index']
    val=val.reset_index()
    del val['index']
    #train = data.sample(frac=0.8,random_state=2017)#keep the nan angle
    #val = data[~data.isin(train)]
        
        
    band_1_tr = np.concatenate([im for im in train['band_1']]).reshape(-1, 75, 75)
    band_2_tr = np.concatenate([im for im in train['band_2']]).reshape(-1, 75, 75)
    full_img_tr = np.stack([band_1_tr, band_2_tr,(band_1_tr+band_2_tr)/2], axis=1)
    angle_tr=train['inc_angle']
    size_tr=train['s1']
    
    band_1_val = np.concatenate([im for im in val['band_1']]).reshape(-1, 75, 75)
    band_2_val = np.concatenate([im for im in val['band_2']]).reshape(-1, 75, 75)
    full_img_val = np.stack([band_1_val, band_2_val,(band_1_val+band_2_val)/2], axis=1)
    angle_val=val['inc_angle']
    size_val=val['s1']

    train_dataset = read_data(data=full_img_tr, labels=train['is_iceberg'],angle=angle_tr,size=size_tr,
                                             transform=transforms.Compose([
                                                   
                                                   RandomHorizontalFlip(),
                                                   RandomVerticallFlip(),
                                                   ToTensor(),

                                               ]))
    
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=128,shuffle=True, num_workers=4)  
    
    val_dataset = read_data(data=full_img_val, labels=val['is_iceberg'],angle=angle_val,size=size_val,
                                             transform=transforms.Compose([
                                                   ToTensor(),
                                               ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=8,shuffle=False, num_workers=4)  


    model_vgg = models.vgg19_bn(pretrained=True)
    model_ft=Vgg11bnNet()

    from itertools import ifilter
    op_parameters = ifilter(lambda p: p.requires_grad, model_ft.parameters())
      
    
    if use_gpu:
        model_ft = model_ft.cuda()
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer_conv = optim.Adam(op_parameters, lr=0.001, betas=(0.9, 0.99))
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    
    new_model,best_loss,best_acc = train_model(model_ft, criterion,optimizer_conv,
                             exp_lr_scheduler1, num_epochs=50)
    
    snapshot_path='.../iceberg/weight/vgg19bnfcn_10cv1/'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path) 
    
    torch.save(model_ft, snapshot_path+'vggbnw_fcn_50_%d_%.4f_%.4f.pkl' % (i,best_loss,best_acc)) 
    
    
    ###train
    val_result=predict(new_model,val_loader,val_dataset,'val')
    assert(val_result[1]==best_loss)
        
    ####test
    test_dataset = read_data(data=full_img_test, labels=test['is_iceberg'],angle=angle_test,size=size_test,
                                             transform=transforms.Compose([
                                                   ToTensor(),
                                               ]))
    
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=8,shuffle=False, num_workers=4) 
    test_result=predict(new_model,test_loader,test_dataset,'test')
 
    test_id=test['id']   
    truth=pd.DataFrame(test_result, columns=['is_iceberg'])
    frame=[test_id,truth]
    result=pd.concat(frame,axis=1)
    
    snapshot_csv='.../iceberg/result/vgg19bnfcn_10cv1/'
    if not os.path.exists(snapshot_csv):
        os.makedirs(snapshot_csv) 
    output=snapshot_csv+'vggbnw_fcn_%d.csv' %i
    result.to_csv(output,index=False) 

    i=i+1