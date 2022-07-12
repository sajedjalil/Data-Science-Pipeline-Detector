import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm_notebook
import cv2
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms
import torch.nn.functional as F


def mask2rle(img):
    #print('img',img)
    rle = []
    if np.all(img == 0):
        return " ".join(rle)
    else:
        tmp = np.rot90( np.flipud( img ), k=3 )
        #print('tmp',tmp)
        lastColor = 0;
        startpos = 0
        endpos = 0
        tmp = tmp.reshape(-1,1) 
        #print('reshape',tmp)
        for i in range( len(tmp) ):
            if (lastColor==0) and tmp[i]>0:
                startpos = i
                lastColor = 1
            elif (lastColor==1)and(tmp[i]==0):
                endpos = i-1
                lastColor = 0
                rle.append( str(startpos)+' '+str(endpos-startpos+1) )
        return " ".join(rle)

    
def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    if rle==0:
        return mask.reshape(imgshape)
    else:        
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        #current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            #current_position += lengths[index]
        return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )

class ImageData(Dataset):
    def __init__(self, df, transform, subset="train"):
        super().__init__()
        self.df = df
        self.transform = transform
        self.subset = subset

        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'

    def __len__(self):
        return int(len(self.df) / 4)

    def __getitem__(self, index):
        fn = self.df['ImageId_ClassId'].iloc[4 * index].split('_')[0]
        fn1 = self.df['ImageId_ClassId'].iloc[4 * index + 3].split('_')[0]
        if fn == fn1:
            img = Image.open(self.data_path + fn)
            img = self.transform(img)
            if self.subset == 'train':
                mask1 = rle2mask(self.df['EncodedPixels'].iloc[4 * index], (256, 1600))
                mask2 = rle2mask(self.df['EncodedPixels'].iloc[4 * index + 1], (256, 1600))
                mask3 = rle2mask(self.df['EncodedPixels'].iloc[4 * index + 2], (256, 1600))
                mask4 = rle2mask(self.df['EncodedPixels'].iloc[4 * index + 3], (256, 1600))
                mask1 = transforms.ToPILImage()(mask1)
                mask1 = self.transform(mask1)
                mask2 = transforms.ToPILImage()(mask2)
                mask2 = self.transform(mask2)
                mask3 = transforms.ToPILImage()(mask3)
                mask3 = self.transform(mask3)
                mask4 = transforms.ToPILImage()(mask4)
                mask4 = self.transform(mask4)
                masks = torch.cat((mask1, mask2, mask3, mask4), 0)
                masks[masks > 0] = 1
                return img, masks
            else:
                mask = None
                img = Image.open(self.data_path + fn)
                img = self.transform(img)
                return img
        else:
            print("alignment error!")

        
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = models.resnet18()
        self.base_model.load_state_dict(torch.load("../input/resnet18/resnet18.pth"))
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        out = self.sigmoid(out)
        
        return out

        
path = '../input/severstal-steel-defect-detection/'       
data_transf = transforms.Compose([
                                  transforms.Scale((256, 256)),
                                  transforms.ToTensor()])

model = UNet(n_class=4).cuda()
model.load_state_dict(torch.load("../input/checkpoint2/v340epoch.pth",map_location='cpu'))

submit = pd.read_csv(path + 'sample_submission.csv', converters={'EncodedPixels': lambda e: ' '})
sub4 = submit[0:6000]

test_data = ImageData(df = sub4, transform = data_transf, subset="test")
test_loader = DataLoader(dataset = test_data, shuffle=False)
 
predict = []
model.eval()

for data in test_loader:
    data = data.cuda()
    output = model(data)
    output = output.cpu().detach().numpy()
    predict.append(abs(output[0]))

pred_rle = []

for p in predict:
    img = np.copy(p)
    for i in range(4):
        tmpimg = img[i]
        mn = 0.8
        tmpimg[tmpimg<=mn] = 0
        tmpimg[tmpimg>mn] = 1     
        tmpimg = cv2.resize(tmpimg, (1600, 256))
        pred_rle.append(mask2rle(tmpimg))

              
submit['EncodedPixels'][0:6000] = pred_rle
submit.to_csv('submission.csv', index=False)

