#https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/145219

import os
import multiprocessing

import sys
import numpy as np
import pandas as pd
import cv2
import skimage.io
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import warnings
warnings.filterwarnings('ignore')
print('torch version:', torch.__version__)
print('\ttorch.cuda.get_device_properties() = %s' % torch.cuda.get_device_properties(0))
print('\tcpu_count = %d' % multiprocessing.cpu_count())
print('\tram = %d MB' % (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') /(1024.**3)))
print('')
# setup---

#local
if 0:
    DATA_DIR = '/root/share1/kaggle/2020/panda/data'
    ADD_DIR  = '/root/share1/kaggle/2020/panda/code/dummy_02/kernel/k01'


# kaggle
if 1:
    DATA_DIR = '../input/prostate-cancer-grade-assessment'
    ADD_DIR  = '../input/panda00'




sys.path.append(ADD_DIR)
from resnext_model import Net as resnext50_net
 
print('import ok!')
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

#### dataset #######################################################################

def make_patch(image, patch_size, num_patch):
    h,w = image.shape[:2]
    s = patch_size

    pad_x = int( patch_size*np.ceil(w/patch_size)-w )
    pad_y = int( patch_size*np.ceil(h/patch_size)-h )
    image = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h,w = image.shape[:2]

    patch = image.reshape(h//s,s,w//s,s,3)
    patch = patch.transpose(0,2,1,3,4).reshape(-1,s,s,3)

    n = len(patch)
    index = np.argsort(patch.reshape(n,-1).sum(-1))[:num_patch]

    y = s*(index//(w//s))
    x = s*(index%(w//s))
    coord = np.stack([x,y,x+s,y+s]).T

    patch = patch[index]
    if len(patch)<num_patch:
        n = num_patch-len(patch)
        patch = np.concatenate([patch, np.full((n,patch_size,patch_size,3),255,dtype=np.uint8)],0)
        coord = np.concatenate([coord, np.full((n,4),-1)],0)
    return patch, coord
class PandaDataset(Dataset):
    def __init__(self, image_dir, image_id):
        self.image_dir = image_dir
        self.image_id  = image_id
    def __len__(self):
        return len(self.image_id)
    def __getitem__(self, index):
        image = skimage.io.MultiImage(self.image_dir + '/%s.tiff'%self.image_id[index])[1]
        patch, coord = make_patch(image, patch_size=256, num_patch=12)

        input = patch.astype(np.float32) / 255
        input = input.transpose(0, 3, 1, 2)
        input = np.ascontiguousarray(input)
        return input



###################################################################################
def run_submit(mode):

    if mode == 'debug':
        time_unit='sec'
        b_print = 5
        image_id, truth = np.array([
            'ac9194d5fe150505d176ca4b96b4be0b',3,
            'fa1a79a5248bf5f5742fb14dabc070c6',3,
            'a66b818d221a0cb079b4b8945a8bacb1',1,
            '3f3ea480b6f8748d6e826f1c1d6474e3',5,
            '29ff1c9442a8be9179e198c0c82f2623',5,
            '58381ead397bcae5aaf3572455872b3e',5,
            '1a3d0f31cb39106c08c2ee02c36f0348',2,
            '7bb1820c1aa24e37399fdf647a1ed64a',1,
            '8eb6325c279ad391cf4a54492db4c41a',4,
            'ff5130c98c14c90ae7f4dd64c297070c',0,
        ]).reshape(-1,2).T
        truth = truth.astype(int)
        image_dir = DATA_DIR + '/train_images'

    if mode == 'debug_time':
        time_unit='min'
        b_print = 10
        df = pd.read_csv(DATA_DIR + '/train.csv')
        image_id = df.image_id.values[:1000]
        truth = df.isup_grade.values[:1000]
        image_dir = DATA_DIR + '/train_images'

    if mode == 'submit':
        time_unit='min'
        b_print = 100
        df = pd.read_csv(DATA_DIR + '/test.csv')
        image_id = df.image_id.values
        image_dir = DATA_DIR + '/test_images'


    #---
    if not os.path.exists(image_dir):
        # submit sample_submission.csv
        df = pd.read_csv(DATA_DIR + '/sample_submission.csv')
        df.to_csv("submission.csv", index=False)

    else:
        dataset = PandaDataset(image_dir, image_id)
        loader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=10,
            drop_last=False,
            num_workers=2,
            pin_memory=True,
        )


        net = []
        for constructor, checkpoint in np.array([
            resnext50_net, 'model/fold0_00031000_model.pth',
            resnext50_net, 'model/fold1_00014000_model.pth',
            resnext50_net, 'model/fold2_00023000_model.pth',
        ]).reshape(-1,2):
            n = constructor().eval().cuda()
            n.load_state_dict(torch.load(ADD_DIR + '/' + checkpoint, map_location=lambda storage, loc: storage),strict=True)
            net.append(n)

        print('load model ok!')

        #---
        num = 0
        predict = []
        probablity = []
        with torch.no_grad():
            start_timer = timer()
            for b, input in enumerate(loader):
                input = input.cuda()
                num += len(input)

                p = 0
                num_augment = 0
                for n in net:
                    for f in [
                        lambda x: x,
                        lambda x: x.flip(-1),
                        lambda x: x.flip(-2),
                        lambda x: x.flip(-1, -2),
                        lambda x: x.transpose(-1, -2),
                        lambda x: x.transpose(-1, -2).flip(-1),
                        lambda x: x.transpose(-1, -2).flip(-2),
                        lambda x: x.transpose(-1, -2).flip(-1, -2),
                    ]:
                        logit = n(f(input.clone()))
                        p += F.softmax(logit, 1)
                        num_augment += 1

                p = p / num_augment
                t = torch.argmax(p,-1)
                probablity.append(p.data.cpu().numpy())
                predict.append(t.data.cpu().numpy())

                if b % b_print==0 or b == len(loader)-1:
                    print(b, input.shape, num, time_to_str(timer() - start_timer, time_unit))
        #---
        probablity = np.concatenate(probablity)
        predict = np.concatenate(predict).reshape(-1)
        print('')

        df_submit = pd.DataFrame({'image_id': image_id, 'isup_grade': predict})
        df_submit.to_csv('submission.csv', index=False)
        print(df_submit.head())
        print('')

        if mode == 'debug':
            print('image_id:', len(image_id))
            print(image_id[:10])
            print('probablity:', probablity.shape)
            print(np.array_str(probablity[:10], precision=5))
            print('predict:', predict.shape)
            print(predict[:10])

            accuracy = (truth==predict).mean()
            print('accuracy',accuracy)

    print('SUCESS!!!')
'''

image_id: 10
['ac9194d5fe150505d176ca4b96b4be0b' 'fa1a79a5248bf5f5742fb14dabc070c6'
 'a66b818d221a0cb079b4b8945a8bacb1' '3f3ea480b6f8748d6e826f1c1d6474e3'
 '29ff1c9442a8be9179e198c0c82f2623' '58381ead397bcae5aaf3572455872b3e'
 '1a3d0f31cb39106c08c2ee02c36f0348' '7bb1820c1aa24e37399fdf647a1ed64a'
 '8eb6325c279ad391cf4a54492db4c41a' 'ff5130c98c14c90ae7f4dd64c297070c']
probablity: (10, 6)
[[1.48840e-04 3.27770e-01 3.09765e-01 3.04353e-01 3.13166e-02 2.66473e-02]
 [2.88053e-02 3.34578e-03 1.02878e-02 6.01481e-01 1.86738e-01 1.69343e-01]
 [1.57316e-04 9.58904e-01 3.71889e-02 1.06949e-04 1.25998e-03 2.38314e-03]
 [2.24930e-04 6.81191e-03 3.04515e-02 2.85657e-01 4.52176e-02 6.31637e-01]
 [5.59224e-01 3.40253e-01 1.66131e-03 8.27511e-02 4.41167e-03 1.16985e-02]
 [2.03036e-03 8.66040e-03 2.92202e-02 2.83767e-02 1.29056e-01 8.02657e-01]
 [6.66314e-03 6.66616e-01 2.39581e-01 3.87415e-02 1.67592e-02 3.16398e-02]
 [9.84212e-05 9.62401e-01 3.43335e-02 1.36423e-03 1.20392e-03 5.98762e-04]
 [4.73687e-01 3.31658e-02 1.44636e-02 1.62525e-01 9.09041e-02 2.25254e-01]
 [9.91780e-01 6.72382e-03 5.46326e-04 1.80501e-04 6.62995e-04 1.06653e-04]]
predict: (10,)
[1 3 1 5 0 5 1 1 0 0]
accuracy 0.6

'''
######################################
#run_submit(mode = 'debug')
#run_submit(mode = 'debug_time')
run_submit(mode = 'submit')