# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import PIL
from sklearn.model_selection import train_test_split
import warnings
from torch.autograd import Variable
warnings.filterwarnings('ignore')

train_path='../input/train/train/'
test_path='../input/test/test/'

labels=pd.read_csv('../input/train.csv')
sub=pd.read_csv('../input/sample_submission.csv')
image_list=os.listdir(train_path)
#plt.imshow(plt.imread(train_path+image_list[0]))
plt.imshow(PIL.Image.open(train_path+image_list[0]))

#image data size =[3,32,32]
class Mydataset(Dataset):
    def __init__(self,label_df,data_path,tranform=None):
        super().__init__()
        self.df=label_df.values
        self.data_dir=data_path
        self.tranform=tranform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__ (self,index):
        image_name,label=self.df[index]
        image_path=os.path.join(self.data_dir,image_name)
        image=PIL.Image.open(image_path).convert('RGB')
        if self.tranform is not None:
            image=self.tranform(image)
        return image,label
   
from torchvision import transforms
tranform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    
test_tranform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

train_label,val_label=train_test_split(labels,test_size=0.1,random_state=425)
train_data=Mydataset(train_label,train_path,tranform)
val_data=Mydataset(val_label,train_path,test_tranform)
test_data=Mydataset(sub,test_path,test_tranform)

train_loader=DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)
val_loader=DataLoader(dataset=val_data,batch_size=64,shuffle=False,num_workers=0)
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=0)

class _CNN(nn.Module):
    def __init__(self):
        super(_CNN,self).__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        
        self.conv2=nn.Sequential(
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv3=nn.Sequential(
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv4=nn.Sequential(
                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        
        
        self.fc=nn.Sequential(
                nn.Linear(256*2*2,1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(1024,2))
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x
cnn=_CNN()
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adagrad(cnn.parameters(),lr=0.01)
epochs=16
loss_lst=[]
accuracy_lst=[]
iteration=[]
count=0
lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=6,gamma=0.5)

for epoch in range(epochs):
    lr_scheduler.step()
    for step,(images,target) in enumerate(train_loader):
        x=Variable(images)
        y=Variable(target)
        
        output=cnn(x)
        loss=loss_function(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
        if (step +1) % 100 ==0:
            correct=0
            totle=0
            for images_val,val_y in val_loader:              
                output=cnn(images_val)
                pred_val= torch.max(output.data, 1)[1]
                correct += (pred_val==val_y).sum().numpy()
                totle += len(val_y)
                accuracy=correct/float(totle)*100
            print('epoch:',epoch,'|step:',step+1,'|Loss:',loss.data.numpy(),'|val_accuracy:',accuracy)
            
            iteration.append(count)
            accuracy_lst.append(accuracy)
            loss_lst.append(loss.data.numpy())


plt.figure()
plt.title('loss curve')
plt.plot(iteration,loss_lst,'b-')
plt.show()

plt.figure()
plt.title('Accuracy curve')
plt.plot(iteration,accuracy_lst,'r-')
plt.show()       


cnn.eval()

preds = []
for batch_i, (data, target) in enumerate(test_loader):
    output = cnn(data)

    pr = output[:,1].detach().numpy()
    for i in pr:
        preds.append(i)

sub['has_cactus'] = preds
sub.to_csv('sub123.csv', index=False)