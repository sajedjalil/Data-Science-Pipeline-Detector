import numpy as np
import pandas as pd
import os
import glob

from PIL import Image

import torch 
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trans

train_images = list(glob.iglob('../input/train/*.jpg'))
test_images = list(glob.iglob('../input/test/*.jpg'))


transforms = trans.Compose([
    trans.Resize((64, 64)),
    trans.ToTensor()
])

class DogCatDataset:
    def __init__(self, images, train=True, transform=None):
        df = pd.DataFrame({'path': images})
        if train:
            df['label'] = df.path.map(lambda x: x.split('/')[-1].split('.')[0])
            df['id'] = df.path.map(lambda x: x.split('/')[-1].split('.')[1])
        else:
            df['id'] = df.path.map(lambda x: x.split('/')[-1].split('.')[0])

        self.train = train
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        img = Image.open(self.df.at[i, 'path'])
        
        if self.train:
            y = 1 if self.df.at[i, 'label'] == 'dog' else 0
        else:
            y = self.df.at[i, 'id'].astype('str')
        
        if self.transform:
            img = self.transform(img)
        return img, y

train_dataset = DogCatDataset(train_images, transform=transforms)
test_dataset = DogCatDataset(test_images, transform=transforms)

train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dl = DataLoader(test_dataset, batch_size=128, num_workers=4)

class DCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.p1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.p2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.p3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.p4 = nn.MaxPool2d(2)
        
        self.l1 = nn.Linear(128 * 4 * 4, 2048)
        self.l2 = nn.Linear(2048, 512)
        self.l3 = nn.Linear(512, 64)
        self.l4 = nn.Linear(64, 1)
    
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.p1(self.conv1(x)))
        x = self.relu(self.p2(self.conv2(x)))
        x = self.relu(self.p3(self.conv3(x)))
        x = self.relu(self.p4(self.conv4(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.sig(self.l4(x))
        return x

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DCClassifier()
model = model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

def train(e):
    model.train()
    l = gl = a = n = 0
    for i, (X, Y) in enumerate(train_dl):
        
        X = X.to(device)
        Y = Y.float().to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, Y.view_as(preds))
        loss.backward()
        optimizer.step()
        
        l += loss.item()
        gl += loss.item()
        a += ((preds.squeeze() > 0.5) == Y.byte()).sum().item() / Y.shape[0]
        n += 1
        if (i+1) % 100 == 0:
            print(f"Epoch {e} Iter {i+1} Loss {l/100} Acc {a/100}")
            l = a = 0
    return gl/n

gl = 99
for i in range(5):
    train(i+1)
