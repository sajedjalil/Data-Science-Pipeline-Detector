#!/usr/bin/python3
# attack_fgsm.py
# A simple demo of Fast Gradient Sign Method in PyTorch.
# This code has been tested with PyTorch 0.1.12.
# The output images of this script is 224x224, so it could not be used in this competition.
# 'dataroot/' is the folder of the dev dataset, including *images.csv* and *images*.
# 
# Copyright 2017 Mengxiao Lin(linmx0130@gmail.com)

import torch
import os
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import argparse

#============
# Data loader of NIPS 2017 Adversarial Example Challenge
import PIL
import numpy as np
import csv
import torchvision.transforms as transforms

# data root: the folder of the dev dataset
DATA_ROOT = './dataroot/'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
imagenet_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
imagenet_transform_wo_normalize = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# simply for images.csv in 
# https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set
def load_ground_truth(csv_filename):
    ret = []
    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            ret.append( (row['ImageId'], int(row['TrueLabel'])) )
    return ret

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, index):
        fid, target = self.data_list[index]
        target = int(target)
        im = PIL.Image.open(os.path.join(DATA_ROOT, 'images', fid+'.png'))
        if self.transform is not None:
            im = self.transform(im)
        return im, target - 1

    def __len__(self):
        return len(self.data_list)
#====================

class NormalizeAtAxies:
    def __init__(self, mean, std, axis):
        self.mean = mean
        self.std = std
        self.axis = axis

    def __call__(self, data):
        for i in range(len(self.mean)):
            data[:, i].sub_(self.mean[i]).div_(self.std[i])
        return data

def clip(d, min_value, max_value):
    idx = d < min_value
    d[idx] = min_value
    idx = d > max_value
    d[idx] = max_value
    return d
    
# set model: cross model attack
imodel = models.resnet152(pretrained=True)
imodel.eval()
imodel_d = models.vgg16(pretrained=True)
imodel_d.eval()
imodel.cuda()
imodel_d.cuda()

def parse_argument():
    parser = argparse.ArgumentParser(description="Attack the ImageNet model")
    parser.add_argument('diff_limit', type=str, help='difference limit of the attack images')
    return parser.parse_args()
args = parse_argument()

normalize = NormalizeAtAxies(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225], axis=1)

diff_limit = int(args.diff_limit) / 255

# get dataset without normalization
val_data_loader = torch.utils.data.DataLoader(
    ListDataset(
        load_ground_truth(os.path.join(data_loader.DATA_ROOT, 'images.csv')),
        transform=data_loader.imagenet_transform_wo_normalize),
    batch_size=32,
    num_workers=1,
    pin_memory=True
)

total = 0
top1 = 0
top1_under_attack = 0
loss_func = nn.CrossEntropyLoss()

for i, (ox, y) in enumerate(val_data_loader):
    total += ox.size()[0]
    x = Variable(normalize(ox.cuda()), requires_grad=True)
    y = Variable(y.cuda())

    f = imodel(x)

    #attack
    imodel.zero_grad()
    loss = loss_func(f, y)
    loss.backward()

    x2 = clip((ox + diff_limit * torch.sign(x.grad.data.cpu())), 0, 1)
    x2 = Variable(normalize(x2.cuda()))
    x = Variable(normalize(ox.cuda()))

    f2 = imodel_d(x)
    _, predicted = torch.max(f2, 1)
    top1 += (predicted.data.cpu() == y.data.cpu()).sum()

    f2 = imodel_d(x2)
    _, predicted = torch.max(f2, 1)
    top1_under_attack += (predicted.data.cpu() == y.data.cpu()).sum()
    print("Progress: {}".format(i))

print("Top 1 Accuracy: {0:.3f}%".format(top1/total*100))
print("Attacked: {0:.3f}%".format(top1_under_attack/total*100))