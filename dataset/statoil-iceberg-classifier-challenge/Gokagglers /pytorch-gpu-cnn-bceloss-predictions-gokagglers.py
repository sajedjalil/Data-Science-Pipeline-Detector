# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from __future__ import print_function
from __future__ import division


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


data = pd.read_json(BASE_FOLDER + '/train.json')

# Suffle
import random
from datetime import datetime
random.seed(datetime.now())
# np.random.seed(datetime.now())
from sklearn.utils import shuffle
data = shuffle(data) # otherwise same validation set each time!
data= data.reindex(np.random.permutation(data.index))

data = shuffle(data) # otherwise same validation set each time!
data= data.reindex(np.random.permutation(data.index))

data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')


band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
full_img = np.stack([band_1, band_2], axis=1)


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
        return self.full_ds[i+self.offset]
    
validationRatio=0.11    

def trainTestSplit(dataset, val_share=validationRatio):
    val_offset = int(len(dataset)*(1-val_share))
    print ("Offest:" + str(val_offset))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, 
                                                                              val_offset, len(dataset)-val_offset)


# In[25]:


batch_size=128

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

num_epoches = 55
dropout = [0.65, 0.55, 0.30, 0.20, 0.10, 0.05]
import math

dropout = torch.nn.Dropout(p=0.30)
relu=torch.nn.LeakyReLU()
pool = nn.MaxPool2d(2, 2)

class ConvRes(nn.Module):
    def __init__(self,insize, outsize):
        super(ConvRes, self).__init__()
        drate = .3
        self.math = nn.Sequential(
                 nn.BatchNorm2d(insize),
                 nn.Dropout(drate),
                 torch.nn.Conv2d(insize, outsize, kernel_size=2,padding=2),
                 nn.PReLU(),
                )
        
    def forward(self, x):
        return self.math(x) 

class ConvCNN(nn.Module):
    def __init__(self,insize, outsize, kernel_size=7, padding=2, pool=2, avg=True):
        super(ConvCNN, self).__init__()
        self.avg=avg
        self.math = torch.nn.Sequential(
            torch.nn.Conv2d(insize, outsize, kernel_size=kernel_size,padding=padding),
            torch.nn.BatchNorm2d(outsize),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(pool,pool),
        )
        self.avgpool=torch.nn.AvgPool2d(pool,pool)
        
    def forward(self, x):
        x=self.math(x)
        if self.avg is True:
            x=self.avgpool(x)
        return x   
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()        
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.cnn1 = ConvCNN (2,32,  kernel_size=7, pool=4, avg=False)
        self.cnn2 = ConvCNN (32,32, kernel_size=5, pool=2, avg=True)
        self.cnn3 = ConvCNN (32,32, kernel_size=5, pool=2, avg=True)
        
        self.res1 = ConvRes (32,64)
        
        self.features = nn.Sequential( 
            self.cnn1,dropout,          
            self.cnn2,
            self.cnn3,
            self.res1,
        )        
        
        self.classifier = torch.nn.Sequential(
            nn.Linear(1024, 1),             
        )
        self.sig=nn.Sigmoid()        
            
    def forward(self, x):
        x = self.features(x) 
        x = x.view(x.size(0), -1)        
        x = self.classifier(x)                
        x = self.sig(x)
        return x        

model = Net()
print(model)

# # Loss and optimizer

# In[28]:


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


# if __name__ == '__main__':

for epoch in range(num_epoches):
    print('Epoch {}'.format(epoch + 1))
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