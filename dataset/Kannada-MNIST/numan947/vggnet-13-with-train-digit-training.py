# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split


np.random.seed(947)
torch.manual_seed(947)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[2]:


train_data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
dig_mnist_data = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
sample_submission = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")

train_data = pd.concat([train_data, dig_mnist_data])


# In[24]:


train_set, valid_set = train_test_split(train_data, random_state=947)

def convert_dataset(data):
    images, labels = [], []
    for idx, row in data.iterrows():
        labels.append([int(row[0])])
        images.append(1.0*row[1:].values.astype(np.float)/255.0)
    return images, labels


# In[25]:


train_images, train_labels = convert_dataset(train_set)
valid_images, valid_labels = convert_dataset(valid_set)
test_images, test_labels = convert_dataset(test_data)
# dig_images, dig_labels = convert_dataset(dig_mnist_data)


# In[26]:


tensor_train_images = torch.stack([torch.Tensor(i.reshape(28,28)) for i in train_images])
tensor_train_labels = torch.stack([torch.LongTensor(i) for i in train_labels])

tensor_valid_images = torch.stack([torch.Tensor(i.reshape(28,28)) for i in valid_images])
tensor_valid_labels = torch.stack([torch.LongTensor(i) for i in valid_labels])

tensor_test_images = torch.stack([torch.Tensor(i.reshape(28,28)) for i in test_images])
tensor_test_indice = torch.stack([torch.Tensor(i) for i in test_labels])

# tensor_dig_images = torch.stack([torch.Tensor(i.reshape(28,28)) for i in dig_images])
# tensor_dig_labels = torch.stack([torch.Tensor(i) for i in dig_labels])


# In[27]:


# Code credit: https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset/55593757
class CustomTensorDataset(data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


# In[28]:


## Coding Credit: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# In[29]:


txf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# In[30]:


train_dataset = CustomTensorDataset((tensor_train_images.view(-1,28,28), tensor_train_labels.reshape(-1).long()), transform=txf)
valid_dataset = CustomTensorDataset((tensor_valid_images.view(-1,28,28), tensor_valid_labels.reshape(-1).long()), transform=txf)
test_dataset = CustomTensorDataset((tensor_test_images.view(-1,28,28), tensor_test_indice.reshape(-1).long()), transform=txf)
# dig_dataset = CustomTensorDataset((tensor_dig_images.view(-1,28,28), tensor_dig_labels.reshape(-1).long()), transform=txf)


# In[31]:


trainloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
validloader = data.DataLoader(valid_dataset, batch_size=128)
testloader = data.DataLoader(test_dataset, batch_size=1)
# digloader = data.DataLoader(dig_dataset, batch_size=128)


# In[32]:


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


# In[33]:


def train(interval, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    train_loss=0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = F.nll_loss(output, target)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    train_loss/=len(train_loader.dataset)
    print('\nTrain Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


# In[34]:


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 512)
        self.fc11 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn11 = nn.BatchNorm1d(256)
        self.cbn1 = nn.BatchNorm2d(20)
        self.cbn2 = nn.BatchNorm2d(50)

    def forward(self, x):
        x = F.relu(self.cbn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.cbn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(F.relu(self.fc11(x)))
        return F.log_softmax(x, dim=1)


# In[35]:


# EPOCHS = 20

# model = BasicNet().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# for epoch in range(1, EPOCHS+ 1):
#     curT = time.time()
#     print("Epoch: {} Learning Rate:  {:0.4f}".format(epoch, scheduler.get_lr()[0]))
#     train(100, model, device, trainloader, optimizer, epoch)
#     test(model, device, validloader)
#     print("Time Taken: {}\n\n".format(time.time()-curT))


# In[36]:


class LeNet(nn.Module):
        def __init__(self):
            super(LeNet,self).__init__()
            self.features=nn.Sequential(
                nn.Conv2d(1,64,3,padding=1),      
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                
                nn.Conv2d(64,64,3,padding=1),        
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                
                nn.Conv2d(64,128,3,padding=1),       
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                
                nn.Conv2d(128,128,3,padding=1),       
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                
                nn.MaxPool2d(2,2),         
                nn.Dropout(p=0.25),
                
                nn.Conv2d(128,256,3,padding=1),        
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                
                nn.Conv2d(256,256,3,padding=1),        
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                
                nn.MaxPool2d(2,2),        
                nn.Dropout(p=0.25),
                
                nn.Conv2d(256,512,3,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                
                nn.Conv2d(512,512,3,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                
                nn.MaxPool2d(2,2),        
                nn.Dropout(p=0.25),
            )
            
            self.classify=nn.Sequential(
                nn.Linear(512*3*3,1024),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.Linear(512,10),
                nn.LogSoftmax(dim=1)
            )
        def forward(self,input):
            x=self.features(input)
            x=x.view(x.size(0),-1)
            x=self.classify(x)
            return x


# In[37]:


# EPOCHS = 50
# PRINT_INTERVAL = 100

# early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.00001)
# model = LeNet().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
# scheduler = StepLR(optimizer,step_size=7, gamma=0.7)


# for epoch in range(1, EPOCHS+ 1):
#     curT = time.time()
#     print("Epoch: {} Learning Rate:  {:0.4f}".format(epoch, scheduler.get_lr()[0]))
#     train(PRINT_INTERVAL, model, device, trainloader, optimizer, epoch)
#     val_loss = test(model, device, validloader)
#     print("Time Taken: {}\n\n".format(time.time()-curT))
#     scheduler.step()
#     early_stopping(val_loss, model)
    
#     if early_stopping.early_stop:
#         break
# #     if args.save_model:
# #         torch.save(model.state_dict(), "mnist_cnn.pt")


# In[38]:


# model.load_state_dict(torch.load("checkpoint.pt"))


# In[39]:


# model


# In[40]:


# config example:
VGG11 = [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']
VGG13 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
VGG16 =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

def vgg_block(config, in_channels):
    layers = []
    for x in config:
        if x == 'M':
            layers.append(nn.MaxPool2d(2,2))
        else:
            layers.append(nn.Conv2d(in_channels, x,3,padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = x
    return nn.Sequential(*layers)

class VGGNet(nn.Module):
    def __init__(self, vgg_config, in_channel, num_classes):
        super(VGGNet, self).__init__()
        self.vgg = vgg_block(vgg_config, in_channel)
        self.adjustor = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Linear(512*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        x = self.vgg(x)
        x = self.adjustor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[ ]:


EPOCHS = 50
PRINT_INTERVAL = 100


model = VGGNet(VGG13, 1, 10).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = StepLR(optimizer,step_size=7, gamma=0.7)
early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.00001)


for epoch in range(1, EPOCHS+ 1):
    curT = time.time()
    print("Epoch: {} Learning Rate:  {:0.4f}".format(epoch, scheduler.get_lr()[0]))
    train(PRINT_INTERVAL, model, device, trainloader, optimizer, epoch)
    val_loss = test(model, device, validloader)
#     test(model, device, digloader)
    print("Time Taken: {}\n\n".format(time.time()-curT))
    scheduler.step()
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        break
#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")


# In[ ]:


# torch.cuda.empty_cache()
model.load_state_dict(torch.load("checkpoint.pt"))

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


preds = []
model.eval()

with torch.no_grad():
    for X, y in testloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        preds.append(np.argmax(y_pred.cpu().numpy()))


# In[ ]:


sample_submission.label = pd.Series(preds)


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:




