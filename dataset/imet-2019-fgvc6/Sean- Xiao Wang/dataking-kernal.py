#!/usr/bin/env python
# coding: utf-8
import os
import gc
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy import ndimage
from timeit import default_timer as timer
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils.data.sampler import *
from torchvision import transforms
from collections import OrderedDict
from torch.utils import model_zoo
from PIL import Image

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

def se_resnext101_32x4d(num_classes=1000, pretrained=None):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model

class IMetDataSet(Dataset):
    def __init__(self, split, augmentation, mode='train',num_classes=1103):
        super(IMetDataSet, self).__init__()
        self.split = split
        self.ids = split['id'].values
        self.aug = augmentation
        self.mode = mode
        self.num_classes = num_classes
        if self.mode in ['train','valid']:
            self.attribute_ids = split['attribute_ids'].values
        else:
            self.attribute_ids = None
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        img_label = np.zeros(self.num_classes).astype(np.float32)
        if self.mode in ['train','valid']:
            img = cv2.imread(os.path.join('../input/imet-2019-fgvc6/','train',img_id + '.png'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            label_idx = [int(idx) for idx in self.attribute_ids[index].split()]
            img_label[label_idx] = 1.0
        elif self.mode in ['test']:
            img = cv2.imread(os.path.join('../input/imet-2019-fgvc6/','test',img_id + '.png'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img = self.aug(img)
        return img,img_label,img_id
        
    def __len__(self):
        return len(self.split)

class BASEMODEL(nn.Module):
    def __init__(self, num_classes=1103):
        super(BASEMODEL, self).__init__()
        self.num_classes = num_classes
        self.base_model = se_resnext101_32x4d()
        self.base_model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.base_model.last_linear = nn.Sequential(
        nn.Linear(self.base_model.last_linear.in_features, num_classes),
        )   

    def forward(self, x):
        x = self.base_model(x)
        return x

def Padding(img, min_size, fill=0, padding_mode='constant'):
    if img.size[0] < min_size:
        img = TF.pad(img, (min_size - img.size[0], 0), fill, padding_mode)
    if img.size[1] < min_size:
        img = TF.pad(img, (0, min_size - img.size[1]), fill, padding_mode)
    return img


def submit():
    out_dir = '../input/dataking5/'
    checkpoints = [
        'fold_0_epoch_14_loss_10.59964848_cv_0.6043_model.pth',
        'fold_3_epoch_14_loss_11.48813152_cv_0.6021_model.pth',
        'fold_2_epoch_14_loss_11.45006371_cv_0.6007_model.pth',
        #'fold_1_epoch_14_loss_11.52565575_cv_0.6023_model.pth',
        'fold_0_epoch_13_loss_10.64157581_cv_0.6094_model.pth',
        'fold_1_epoch_12_loss_10.00210667_cv_0.6073_model.pth',  
        'fold_4_epoch_14_loss_11.04420853_cv_0.6035_model.pth',
        'fold_2_epoch_13_loss_9.98005390_cv_0.6046_model.pth',
        'fold_4_epoch_14_loss_11.04420853_cv_0.6035_model.pth',
        'fold_3_epoch_13_loss_10.50909519_cv_0.6083_model.pth'
                  ]

    folds = len(checkpoints)
    batch_size = 60
    threshold = 0.12
    num_classes = 1103
    net = BASEMODEL().cuda()
    test_df = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')
    all_probs = np.zeros(shape=(test_df.shape[0],num_classes))
    #test_dataset = IMetDataSet(test_df, test_augment, 'test')
    test_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: Padding(img, min_size=336)),
    transforms.FiveCrop(336),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transforms.ToTensor()(crop)) for crop in crops])),
    ])
    test_dataset = IMetDataSet(test_df, test_augment, 'test')
    for _fold in range(folds):
        print("fold: ",_fold)
        if checkpoints[_fold] is not None:
            print('\tload_checkpoint = %s\n' % checkpoints[_fold])
            net.load_state_dict(torch.load(out_dir + checkpoints[_fold], map_location=lambda storage, loc: storage))
        test_loader  = DataLoader(
                                test_dataset,
                                sampler     = SequentialSampler(test_dataset),
                                batch_size  = batch_size,
                                drop_last   = False,
                                num_workers = 4,
                                pin_memory  = True) 
        all_prob = []
        all_id = []
        for input, truth, image_id in tqdm(test_loader):
            bs, ncrops, c, h, w = input.size()
            input = input.cuda()
            truth = truth.cuda()
            with torch.no_grad():
                logit = data_parallel(net, input.view(-1,c,h,w))
                prob  = F.sigmoid(logit)
                prob  = prob.view(bs, ncrops, -1).mean(1)
            prob = prob.squeeze().data.cpu().numpy()
            all_prob.append(prob)
            all_id.append(image_id)
        all_prob = np.concatenate(all_prob)
        all_probs += all_prob
        all_id = np.concatenate(all_id).tolist()

    all_probs /= folds
    all_pred = []
    for prob in all_probs:
        s = ' '.join(list([str(i) for i in np.nonzero(prob>threshold)[0]]))
        all_pred.append(s)
    sub_df = pd.DataFrame({ 'id' : all_id , 'attribute_ids' : all_pred}).astype(str)
    sub_df.to_csv('submission.csv', header=True, index=False) 

if __name__ == '__main__':
    submit()

