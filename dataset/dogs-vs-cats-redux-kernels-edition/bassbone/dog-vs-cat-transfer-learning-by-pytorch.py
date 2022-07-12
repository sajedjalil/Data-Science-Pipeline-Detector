## import library
import numpy as np
import pandas as pd
import os, glob, time, copy, random, zipfile
from statistics import mean
from PIL import Image
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

os.system('pip install efficientnet_pytorch')
from efficientnet_pytorch import EfficientNet

## const
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32
LR = 1e-4
BASE_DIR = '../input/dogs-vs-cats-redux-kernels-edition'
DATA_DIR = '../data'
TRAIN_DIR = DATA_DIR + '/train'
TEST_DIR = DATA_DIR + '/test'
TEST_RATIO = .1
NUM_EPOCH = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 2020
SPLITS = 5
MODEL_NAME = 'efficientnet-b3'

## class definition
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase):
        return self.data_transform[phase](img)

class DogVsCatDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):    
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        return img_transformed, label

## function
def create_params_to_update(net):
    params_to_update_1 = []
    update_params_name_1 = ['_fc.weight', '_fc.bias']

    for name, param in net.named_parameters():
        if name in update_params_name_1:
            param.requires_grad = True
            params_to_update_1.append(param)
            #print("{} 1".format(name))
        else:
            param.requires_grad = False
            #print(name)

    params_to_update = [
        {'params': params_to_update_1, 'lr': LR}
    ]
    
    return params_to_update

def adjust_learning_rate(optimizer, epoch):
    lr = LR * (0.1**(epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):
    
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc, best_loss = 0.0, float('inf')
    net = net.to(DEVICE)
    
    for epoch in range(num_epoch):
        #print('-'*20)
        print('Epoch {}/{} ---'.format(epoch + 1, num_epoch))
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                net.train()
            else:
                net.eval()
                
            epoch_loss, epoch_corrects = 0.0, 0
            
            for inputs, labels in dataloader_dict[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data).item()
                    
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = float(epoch_corrects) / len(dataloader_dict[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc, best_loss, best_model_wts = epoch_acc, epoch_loss, copy.deepcopy(net.state_dict())
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    print('Best val loss: {:4f}'.format(best_loss))

    net.load_state_dict(best_model_wts)
    return net, best_acc, best_loss
    
def f_train(train_list, val_list):
    train_dataset = DogVsCatDataset(train_list, transform=ImageTransform(SIZE, MEAN, STD), phase='train')
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = DogVsCatDataset(val_list, transform=ImageTransform(SIZE, MEAN, STD), phase='val')
    val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    net = EfficientNet.from_pretrained(MODEL_NAME)
    print(net)
    net._fc = nn.Linear(in_features=1536, out_features=2)

    params_to_update = create_params_to_update(net)
    optimizer = optim.Adam(params=params_to_update, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    net, acc, loss = train_model(net, dataloader_dict, criterion, optimizer, NUM_EPOCH)
    #print(acc)
    #print(loss)
    return net, acc, loss

os.makedirs(DATA_DIR, exist_ok=True)

for file in ['train.zip', 'test.zip']:
    with zipfile.ZipFile(os.path.join(BASE_DIR, file)) as zip_obj:
        zip_obj.extractall(DATA_DIR)

train_list = pd.Series(glob.glob(os.path.join(TRAIN_DIR, '*.jpg')))
test_list = pd.Series(glob.glob(os.path.join(TEST_DIR, '*.jpg')))

print("train:{}".format(len(train_list)))
kf = KFold(n_splits=SPLITS, shuffle=True, random_state=SEED)
nets = []
accs = []
losses = []

for train, valid in kf.split(train_list):
    train_list2 = train_list.iloc[train].reset_index(drop=True)
    valid_list2 = train_list.iloc[valid].reset_index(drop=True)
    net, acc, loss = f_train(train_list2, valid_list2)
    nets.append(net)
    accs.append(acc)
    losses.append(loss)

print("oof acc: {:4f}".format(mean(accs)))
print("oof loss: {:4f}".format(mean(losses)))

## prediction

id_list, pred_list = [], []

with torch.no_grad():
    for test_path in test_list:
        img = Image.open(test_path)
        _id = int(test_path.split('/')[-1].split('.')[0])

        transform = ImageTransform(SIZE, MEAN, STD)
        img = transform(img, phase='val')
        img = img.unsqueeze(0)
        img = img.to(DEVICE)

        pred = 0.0
        for i in range(SPLITS):
            net = nets[i]
            net.eval()

            outputs = net(img)
        
            preds = F.softmax(outputs, dim=1)[:, 1].tolist()
            pred += preds[0]
        
        id_list.append(_id)
        pred_list.append(pred / SPLITS)
    
res = pd.DataFrame({
    'id': id_list,
    'label': pred_list
})

res.sort_values(by='id', inplace=True)
res.reset_index(drop=True, inplace=True)
res.to_csv('submission.csv', index=False)
