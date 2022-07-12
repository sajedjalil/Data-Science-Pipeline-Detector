# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import math

width,height = 768,768

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

class ShipLocationDataset(Dataset):
    def __init__(self,data,path):
        self.data = data 
        self.filtered = data.drop_duplicates('ImageId')
        self.path = path
        
    def __getitem__(self,index):
        index = int(index)
        imageid,encodedpixels = self.filtered.iloc[index]
        records = self.data.loc[self.data['ImageId'] == imageid]
        target = self.generate_target(records)
        img = cv2.imread(self.path + imageid)        
        inputs = transform(img).float()
        return (inputs,target)
        
    def __len__(self):
        return len(self.filtered)

    def generate_target(self,records):
        result = torch.tensor((), dtype=torch.float)
        result = result.new_zeros(height*width)
        for record in records['EncodedPixels']:
            if record=='nan':
                return result.reshape(height,width)
            if type(record) is float:
                if (math.isnan(record)):
                    return result.reshape(height,width)
                else:
                    result[int(record)] = 1
            else:
                #print("Record is: ",type(record))
                rec = np.fromstring(record, dtype=int, sep=' ')
                it = np.nditer(rec)
                while not it.finished:
                    i = it[0]
                    it.iternext()
                    j = it[0]
                    it.iternext()
                    result[i:i+j] = 1.0
        return torch.transpose(result.reshape((height,width)),0,1)

#shipdata = pd.read_csv("train_ship_segmentations.csv")        
#inputs = ShipLocationDataset(shipdata,'./data/train/')