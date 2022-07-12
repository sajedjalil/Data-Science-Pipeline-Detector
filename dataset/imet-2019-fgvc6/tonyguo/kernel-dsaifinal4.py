import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import math
import pandas as pd
from PIL import Image
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import random
import datetime
import os

from sklearn import preprocessing 

train = pd.read_csv("../input/train.csv")
lable = pd.read_csv("../input/labels.csv")
test = pd.read_csv("../input/sample_submission.csv")

lable_length = len(lable)
train_length = len(train)
test_length = len(test)
print(train_length)
print(lable_length)
print(test_length)


def creatData(train,lable_length):
    train = np.array(train)
    train_data = []
    for t in range(train_length):
        v = np.zeros(lable_length)
        #print(train[t,1])
        for s in train[t,1].split(" "):
            #print(s)
            v[int(s)] = 1
        train_data.append([train[t,0],v])
    return np.array(train_data)
    
    
train_lib = creatData(train,lable_length)
#print(train_lib)

train_transformer = transforms.Compose([
  transforms.Resize((128,128)),              # resize the image to 
  #transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
  transforms.ToTensor()])             # transform it into a PyTorch Tensor


class trainDataset(Dataset):
    def __init__(self, train_lib, transform=None):
        self.filenames = train_lib[:,0]
        self.labels = train_lib[:,1]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open("../input/train/"+format(self.filenames[idx])+'.png')  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]
        
class testDataset(Dataset):
    def __init__(self, test_lib, transform=None):
        test_lib = np.array(test_lib)
        self.filenames = test_lib[:,0]
        #self.labels = test_lib[:,1]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open("../input/test/"+format(self.filenames[idx])+'.png')  # PIL image
        image = self.transform(image)
        return image,self.filenames[idx]
        
train_dataloader = DataLoader(trainDataset(train_lib, train_transformer), batch_size=128, shuffle=True)

test_dataloader = DataLoader(testDataset(test, train_transformer),batch_size=128,shuffle=False)

####################################################################
                                          

resnet_model = models.resnet18(pretrained=False) 
resnet_model.fc= nn.Linear(in_features=512, out_features=lable_length)


cnn = resnet_model
cnn.cuda()
print(cnn)

def train(epoch):
    for step, (x, y) in enumerate(train_dataloader):
        data = Variable(x).cuda()   # batch x
        target = Variable(y).cuda()   # batch y
        #print(data.type())
        #print(target.type())
        output = cnn(data)               # cnn output
        #loss = nn.functional.nll_loss(output, target)
        loss = loss_func(output, target.float())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        if step==0:
            start = time.time()
            ti = 0
        elif step==100:
            ti = time.time()-start #total time = ti*(length/100)
            #print(ti)
            ti = ti*(len(train_dataloader)/100)
        if step % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\tTime Remain : {} '.
                     format(epoch, 
                            step * len(data), 
                            len(train_dataloader.dataset),
                            100.*step/len(train_dataloader), 
                            loss.data.item(),
                            datetime.timedelta(seconds=(ti*((int(len(train_dataloader)-step)/len(train_dataloader)))))))
    
    print("Finish")
    
    
for epoch in range(10):
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.005/(2**epoch),momentum=0.9)
    #optimizer = torch.optim.ASGD(cnn.parameters(), lr=0.001)
    #optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001/(2**epoch))
    loss_func = torch.nn.MSELoss()
    #loss_func = torch.nn.MultiLabelMarginLoss()
    #loss_func = torch.nn.SmoothL1Loss()
    #loss_func = FocalLoss(class_num = lable_length)
    #optimizer = torch.optim.ASGD(cnn.parameters(), lr=0.0005/(epoch+1))
    train(epoch)
    na = "net_"+format(epoch)+".pkl"
    torch.save(cnn, na)
    print("nwt saved as : "+na )
    

#torch.save(cnn, 'net.pkl')


def findPre(output):
    a = ''
    output = np.array(output)
    for i in range(len(output)):
        if output[i]>0.95:
            #print(output[i])
            a = a + format(i)+' '
    #print(a)
    return a
    
def test(model):
    model = model.eval()
    #model = model.cpu().eval()
    ans = []
    for step, (x, y) in enumerate(test_dataloader):
        data = Variable(x).cuda()
        #data = Variable(x)
        target = y
        output = model(data)
        v = output.cpu().detach()
        v = torch.sigmoid(v)
        v = np.array(v)
        v = preprocessing.minmax_scale(v, feature_range=(-1,1),axis=1)
        #v = min_max_scaler.fit_transform(v)
#         v = torch.from_numpy(v)
#         v = F.softmax(v, dim=0)
#         v = np.array(v)
        #v = sigmoid(v)
        #print("==========")
        #print(np.max(v[0]))
        #print(np.min(v[0]))
        #print("==========")
        for i in range(len(v)):
            #V = (v[i]+abs(np.min(v[i])))/(abs(np.min(v[i]))+abs(np.max(v[i])))
            #print(v)
            s = findPre(v[i])
            ans.append([target[i],s])
        if step %10 == 0:
            print('[{}/{} ({:.0f}%)]'.format(step * len(data), 
                                        len(test_dataloader.dataset),
                                        100.*step/len(test_dataloader)))
    print("Finish")
    return ans
    
sub = test(cnn)

sub =  pd.DataFrame(sub)

sub = sub.rename(index=str, columns={0: "id", 1: "attribute_ids"})

sub.head

sub.to_csv('submission.csv', index=False)
    






