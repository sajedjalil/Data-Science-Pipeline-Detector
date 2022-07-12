# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
import cv2

#For those struggling with PIL augmention

class ImageDataset(data.Dataset):
    """So you can write custom pytorch dataset. You only have to scpecify 2 functions:
    -  __getitem__
    - __len__
    """
    def __init__(self, X_data, include_target, u = 0.5, X_transform = None):
        """X_data = pandas df
        include_target - flag. include_target = True if train and False if test
        u - arg in X_transform function
        X_transform - your augmentation function
        """
        self.X_data = X_data
        self.include_target = include_target
        self.X_transform = X_transform
        self.u = u

    def __getitem__(self, index):
        np.random.seed() #see comment
        #get 2 channels of our image
        img1 = self.X_data.iloc[index]['band_1']
        img2 = self.X_data.iloc[index]['band_2']

        #image shape = (75,75,2)
        img = np.stack([img1, img2], axis = 2)
        
        #get angle and img_name
        angle = self.X_data.iloc[index]['inc_angle']
        img_id = self.X_data.iloc[index]['id']
        
        #perform augmentation
        if self.X_transform:
            img = self.X_transform(img, **{'u' : self.u})

        #Reshape image for pytorch
        img = img.transpose((2, 0, 1))
        img_numpy = img.astype(np.float32)
        #convert image to tensor
        img_torch = torch.from_numpy(img_numpy)
        
        #so our loader will yield dictionary wi such fields:
        dict_ = {'img' : img_torch,
                'id' : img_id, 
                'angle' : angle,
                'img_np' : img_numpy}
        
        #if train - then also include target
        if self.include_target:
            target = self.X_data.iloc[index]['is_iceberg']
            dict_['target'] = target

        return dict_

    def __len__(self):
        return len(self.X_data)
        
#your custom aug function for numpy image:
#seems like all flip augmentations may decrease performance
def random_vertical_flip(img, u=0.5):
    if np.random.random() < u:
        img = cv2.flip(img, 0)

    return img
        
train = pd.read_json('../input/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
train['band_1'] = train['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
train['band_2'] = train['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
        
batch_size = 10
train_ds = ImageDataset(train, include_target = True, u =0.5, X_transform = random_vertical_flip)
USE_CUDA = False #for kernel
THREADS = 4 #for kernel
train_loader = data.DataLoader(train_ds, batch_size,
                                    sampler = RandomSampler(train_ds),
                                    num_workers = THREADS,
                                    pin_memory= USE_CUDA )
                                    
#prseudo code for train
for i, dict_ in enumerate(train_loader):
    images  = dict_['img']
    target  = dict_['target'].type(torch.FloatTensor)
    
    if USE_CUDA:
        images = images.cuda()
        target = target.cuda()
    
    images = Variable(images)
    target = Variable(target)

    #train net
    #prediction = Net().forward(images)
    #....
    
    #for kernel:
    print(target)
    if i ==0 : break

                                    