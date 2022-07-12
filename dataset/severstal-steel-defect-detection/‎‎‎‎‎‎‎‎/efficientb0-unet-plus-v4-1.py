import os
import sys
from shutil import copyfile
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
print('torch version:', torch.__version__)

#copyfile(src = '../input/steel2019/efficientnet.py', dst = '../working/efficientnet.py')
sys.path.append('../input/steel2019')
from efficientnet import *


#######################################################################
import warnings
warnings.filterwarnings('ignore')


# kaggle kernel
DATA_DIR  = '../input/severstal-steel-defect-detection'
SUBMISSION_CSV_FILE = 'submission.csv'
CHECKPOINT_FILE = '../input/steel2019/00090000_model.pth'


# overwrite to local path
# DATA_DIR = '/root/share/project/kaggle/2019/steel/data'
# SUBMISSION_CSV_FILE = \
#     '/root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1/submit/kernel-submission.csv'
# CHECKPOINT_FILE = \
#         '/root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1/checkpoint/00090000_model.pth'



# etc ############################################################

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

#### data #########################################################

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

def image_to_input(image):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    input[:,0] = (input[:,0]-IMAGE_RGB_MEAN[0])/IMAGE_RGB_STD[0]
    input[:,1] = (input[:,1]-IMAGE_RGB_MEAN[1])/IMAGE_RGB_STD[1]
    input[:,2] = (input[:,2]-IMAGE_RGB_MEAN[2])/IMAGE_RGB_STD[2]
    return input


class KaggleTestDataset(Dataset):
    def __init__(self):

        df =  pd.read_csv(DATA_DIR + '/sample_submission.csv')
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.uid = df['ImageId'].unique().tolist()

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        return string

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        # print(index)
        image_id = self.uid[index]
        image = cv2.imread(DATA_DIR + '/test_images/%s'%(image_id), cv2.IMREAD_COLOR)
        return image, image_id


def null_collate(batch):
    batch_size = len(batch)

    input = []
    image_id = []
    for b in range(batch_size):
        input.append(batch[b][0])
        image_id.append(batch[b][1])

    input = np.stack(input)
    input = torch.from_numpy(image_to_input(input))
    return input, image_id




## -- kaggle --

def post_process(mask, min_size):
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predict = np.zeros((256, 1600), np.float32)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = 1

    return predict


def run_length_encode(mask):

    m = mask.T.flatten()
    if m.sum()==0:
        rle=''
    else:
        start  = np.where(m[1: ] > m[:-1])[0]+2
        end    = np.where(m[:-1] > m[1: ])[0]+2
        length = end-start

        rle = [start[0],length[0]]
        for i in range(1,len(length)):
            rle.extend([start[i],length[i]])

        rle = ' '.join([str(r) for r in rle])

    return rle

##### net ##############################################################

def upsize2(x):
    #x = F.interpolate(x, size=e.shape[2:], mode='nearest')
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d( out_channel//2),
            Swish(), #nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channel//2),
            Swish(), #nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channel),
            Swish(), #nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.top(torch.cat(x, 1))
        return x



class Net(nn.Module):
    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        self.mix = nn.Parameter(torch.FloatTensor(5))
        self.mix.data.fill_(1)

        self.e = EfficientNet(drop_connect_rate)
        self.block = [
            self.e.stem,
            self.e.block1, self.e.block2, self.e.block3, self.e.block4, self.e.block5, self.e.block6, self.e.block7,
            self.e.last
        ]
        self.e.logit = None  #dropped


        self.decode0_1 =  Decode(16+24, 24)

        self.decode1_1 =  Decode(24+40, 40)
        self.decode0_2 =  Decode(16+24+40, 40)

        self.decode2_1 =  Decode(40+112, 112)
        self.decode1_2 =  Decode(24+40+112, 112)
        self.decode0_3 =  Decode(16+24+40+112, 112)

        self.decode3_1 =  Decode(112+1280, 128)
        self.decode2_2 =  Decode(40+112+128, 128)
        self.decode1_3 =  Decode(24+40+112+128, 128)
        self.decode0_4 =  Decode(16+24+40+112+128, 128)

        self.logit1 = nn.Conv2d(24,num_class, kernel_size=1)
        self.logit2 = nn.Conv2d(40,num_class, kernel_size=1)
        self.logit3 = nn.Conv2d(112,num_class, kernel_size=1)
        self.logit4 = nn.Conv2d(128,num_class, kernel_size=1)


    def forward(self, x):
        batch_size,C,H,W = x.shape

        #----------------------------------
        #extract efficientnet feature

        backbone = []
        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

            if i in [1,2,3,5,8]:
                backbone.append(x)


        #----------------------------------
        x0_0 = backbone[0] # 16
        x1_0 = backbone[1] # 24
        x0_1 = self.decode0_1([x0_0, upsize2(x1_0)])

        x2_0 = backbone[2] # 40
        x1_1 = self.decode1_1([x1_0, upsize2(x2_0)])
        x0_2 = self.decode0_2([x0_0, x0_1, upsize2(x1_1)])

        x3_0 = backbone[3] #112
        x2_1 = self.decode2_1([x2_0, upsize2(x3_0)])
        x1_2 = self.decode1_2([x1_0, x1_1, upsize2(x2_1)])
        x0_3 = self.decode0_3([x0_0, x0_1, x0_2, upsize2(x1_2)])


        x4_0 = backbone[4] #1280
        x3_1 = self.decode3_1([x3_0, upsize2(x4_0)])
        x2_2 = self.decode2_2([x2_0, x2_1, upsize2(x3_1)])
        x1_3 = self.decode1_3([x1_0, x1_1, x1_2, upsize2(x2_2)])
        x0_4 = self.decode0_4([x0_0, x0_1, x0_2, x0_3, upsize2(x1_3)])


        # deep supervision
        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        logit4 = self.logit4(x0_4)

        logit = self.mix[1]*logit1 + self.mix[2]*logit2 + self.mix[3]*logit3 + self.mix[4]*logit4
        logit = F.interpolate(logit, size=(H,W), mode='bilinear', align_corners=False)
        return logit  #logit, logit0



#########################################################################

def run_check_setup():

    ## load net
    net = Net().cuda()
    net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=lambda storage, loc: storage))

    ## load data
    image_id = ['004f40c73.jpg', '006f39c41.jpg', '00b7fb703.jpg', '00bbcd9af.jpg']
    image=[]
    for i in image_id:
        m = cv2.imread(DATA_DIR +'/test_images/%s'%i)
        image.append(m)
    image=np.stack(image)
    input = image_to_input(image)
    input = torch.from_numpy(input).cuda()

    #run here!
    net.eval()
    with torch.no_grad():
        logit = net(input)
        probability= torch.sigmoid(logit)

    print('input: ',input.shape)
    print('logit: ',logit.shape)
    print('')
    #---
    input = input.data.cpu().numpy()
    logit = logit.data.cpu().numpy()

    if 1:
        print(logit[0,0,:5,:5],'\n')
        print(logit[3,0,-5:,-5:],'\n')
        print(logit.mean(),logit.std(),logit.max(),logit.min(),'\n')
        print('')
        print('---------------------')
        print('')
        '''
        
torch version: 1.1.0
input:  torch.Size([4, 3, 256, 1600])
logit:  torch.Size([4, 4, 256, 1600])

[[-7.768246  -7.7584023 -7.738714  -7.8852553 -8.198025 ]
 [-7.7995114 -7.8147035 -7.8450875 -8.025144  -8.354872 ]
 [-7.8620415 -7.9273057 -8.057835  -8.304921  -8.668568 ]
 [-7.7528515 -7.8566213 -8.06416   -8.336199  -8.672734 ]
 [-7.471942  -7.60265   -7.864067  -8.118975  -8.367373 ]] 

[[-6.267967  -6.2087927 -6.2660394 -6.4397073 -6.526541 ]
 [-6.415012  -6.3766723 -6.3949575 -6.4698677 -6.5073223]
 [-6.5491033 -6.494913  -6.5103583 -6.59544   -6.6379805]
 [-6.6702423 -6.5635138 -6.6122413 -6.8164234 -6.9185147]
 [-6.7308116 -6.5978146 -6.6631827 -6.9269156 -7.058782 ]] 

-7.79056 1.8560479 3.6647573 -12.449509 
        
        '''


########################################################################

def run_make_submission_csv():

    threshold_pixel = [0.5,0.5,0.6,0.5]
    min_size = [800,1000,3000,3500]

    ## load net
    print('load net ...')
    net = Net().cuda()
    net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=lambda storage, loc: storage))
    print('')


    ## load data
    print('load data ...')
    dataset = KaggleTestDataset()
    print(dataset)
    #exit(0)

    loader  = DataLoader(
        dataset,
        sampler     = SequentialSampler(dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    #test time augmentation  -----------------------
    def null_augment   (input): return input
    def flip_lr_augment(input): return torch.flip(input, dims=[2])
    def flip_ud_augment(input): return torch.flip(input, dims=[3])

    def null_inverse_augment   (logit): return logit
    def flip_lr_inverse_augment(logit): return torch.flip(logit, dims=[2])
    def flip_ud_inverse_augment(logit): return torch.flip(logit, dims=[3])

    augment = (
        (null_augment,   null_inverse_augment   ),
        (flip_lr_augment,flip_lr_inverse_augment),
        (flip_ud_augment,flip_ud_inverse_augment),
    )


    ## start here ----------------------------------
    image_id_class_id = []
    encoded_pixel     = []


    net.eval()

    start = timer()
    for t,(input, image_id) in enumerate(loader):
        print('\r loader: t = 4%d / 4%d  %s  %s : %s'%(
              t, len(loader)-1, str(input.shape), image_id[0], time_to_str((timer() - start),'sec'),
        ),end='', flush=True)

       
        with torch.no_grad():
            input = input.cuda()
            for k, (a, inv_a) in enumerate(augment):
                logit = net(a(input))
                p = inv_a(torch.sigmoid(logit))

                if k ==0:
                    probability  = p**0.5
                else:
                    probability += p**0.5

            probability = probability/len(augment)
            probability = probability.data.cpu().numpy()
            batch_size = len(image_id)
            for b in range(batch_size):
                for c in range(4):
                    predict = probability[b,c]

                    predict = predict>threshold_pixel[c]
                    predict = post_process(predict, min_size[c])
                    rle = run_length_encode(predict)

                    image_id_class_id.append(image_id[b]+'_%d'%(c+1))
                    encoded_pixel.append(rle)


    df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv(SUBMISSION_CSV_FILE, index=False)

    ## print statistics ----
    if 1:
        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        pos1 = ((df['Class']==1) & (df['Label']==1)).sum()
        pos2 = ((df['Class']==2) & (df['Label']==1)).sum()
        pos3 = ((df['Class']==3) & (df['Label']==1)).sum()
        pos4 = ((df['Class']==4) & (df['Label']==1)).sum()

        num_image = len(df)//4
        num = len(df)
        pos = (df['Label']==1).sum()
        neg = num-pos

        print('')
        print('\t\tnum_image = %5d(1801)'%num_image)
        print('\t\tnum  = %5d(7204)'%num)
        print('\t\tneg  = %5d(6172)  %0.3f'%(neg,neg/num))
        print('\t\tpos  = %5d(1032)  %0.3f'%(pos,pos/num))
        print('\t\tpos1 = %5d( 128)  %0.3f  %0.3f'%(pos1,pos1/num_image,pos1/pos))
        print('\t\tpos2 = %5d(  43)  %0.3f  %0.3f'%(pos2,pos2/num_image,pos2/pos))
        print('\t\tpos3 = %5d( 741)  %0.3f  %0.3f'%(pos3,pos3/num_image,pos3/pos))
        print('\t\tpos4 = %5d( 120)  %0.3f  %0.3f'%(pos4,pos4/num_image,pos4/pos))



# main #################################################################
if __name__ == '__main__':
    #run_check_setup()
    run_make_submission_csv()

    print('\nsucess!')