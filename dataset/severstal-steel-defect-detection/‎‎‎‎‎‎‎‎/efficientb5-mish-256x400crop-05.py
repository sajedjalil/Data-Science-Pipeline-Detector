import os
import sys
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



# setup external file ############################################################
#local
if 0:
    MYFILE_DIR = '/root/share/project/kaggle/2019/steel/code/dummy_semi_17/kernel/efficientb5_mish_256x400crop'
    DATA_DIR   = '/root/share/project/kaggle/2019/steel/data'
    CHECKPOINT_FILE     = '/root/share/project/kaggle/2019/steel/result100/efficientb5-mish-fpn-crop256x400-semi-foldb1-1/checkpoint/00097000_model.pth'
    SUBMISSION_CSV_FILE = '/root/share/project/kaggle/2019/steel/result100/efficientb5-mish-fpn-crop256x400-semi-foldb1-1/kernel-submission.csv'


#kaggle
if 1:
    MYFILE_DIR  = '../input'
    DATA_DIR    = '../input/severstal-steel-defect-detection'
    SUBMISSION_CSV_FILE = 'submission.csv'
    CHECKPOINT_FILE     = '../input/myfile05/00097000_model.pth'


#copyfile(src = '../input/steel2019/efficientnet.py', dst = '../working/efficientnet.py')
sys.path.append(MYFILE_DIR)
from myfile05.efficientnet import *
from myfile05.helper import *
import warnings
warnings.filterwarnings('ignore')


#### net #########################################################################
class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, num_group=32, kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.gn   = nn.GroupNorm(num_group,out_channel)

    def forward(self,x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

def upsize_add(x, lateral):
    return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral

def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x


class Net(nn.Module):

    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = EfficientNetB5(drop_connect_rate)
        self.stem   = e.stem
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        self.block5 = e.block5
        self.block6 = e.block6
        self.block7 = e.block7
        self.last   = e.last
        e = None  #dropped

        #---
        self.lateral0 = nn.Conv2d(2048, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral1 = nn.Conv2d( 176, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral2 = nn.Conv2d(  64, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral3 = nn.Conv2d(  40, 64,  kernel_size=1, padding=0, stride=1)

        self.top1 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d( 64, 64),
        )
        self.top4 = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.logit_mask = nn.Conv2d(64,num_class+1,kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        x = self.stem(x)            #; print('stem  ',x.shape)
        x = self.block1(x)    ;x0=x #; print('block1',x.shape)
        x = self.block2(x)    ;x1=x #; print('block2',x.shape)
        x = self.block3(x)    ;x2=x #; print('block3',x.shape)
        x = self.block4(x)          #; print('block4',x.shape)
        x = self.block5(x)    ;x3=x #; print('block5',x.shape)
        x = self.block6(x)          #; print('block6',x.shape)
        x = self.block7(x)          #; print('block7',x.shape)
        x = self.last(x)      ;x4=x #; print('last  ',x.shape)

        # segment
        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3)) #16x16
        t2 = upsize_add(t1, self.lateral2(x2)) #32x32
        t3 = upsize_add(t2, self.lateral3(x1)) #64x64

        t1 = self.top1(t1) #128x128
        t2 = self.top2(t2) #128x128
        t3 = self.top3(t3) #128x128

        t = torch.cat([t1,t2,t3],1)
        t = self.top4(t)
        logit_mask = self.logit_mask(t)
        logit_mask = F.interpolate(logit_mask, scale_factor=2.0, mode='bilinear', align_corners=False)

        return logit_mask


def probability_mask_to_probability_label(probability):
    probability = F.adaptive_max_pool2d(probability,1).squeeze(-1).squeeze(-1)
    return probability


#### data #########################################################################

def image_to_input(image):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    # input[:,0] = (input[:,0]-IMAGE_RGB_MEAN[0])/IMAGE_RGB_STD[0]
    # input[:,1] = (input[:,1]-IMAGE_RGB_MEAN[1])/IMAGE_RGB_STD[1]
    # input[:,2] = (input[:,2]-IMAGE_RGB_MEAN[2])/IMAGE_RGB_STD[2]
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



### kaggle ##############################################################
def post_process(mask, min_size):
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predict = np.zeros((256, 1600), np.float32)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = 1

    return predict


#https://www.kaggle.com/bigkizd/se-resnext50-89
def run_length_encode(mask):
    m = mask.T.flatten()
    if m.sum()==0:
        rle=''
    else:
        m   = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = ' '.join(str(r) for r in run)
    return rle



#########################################################################

def run_check_setup():

    ## load net
    net = Net().cuda()
    net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=lambda storage, loc: storage),strict=True)

    ## load data
    image_id = ['004f40c73.jpg', '006f39c41.jpg', '00b7fb703.jpg', '00bbcd9af.jpg']
    image=[]
    for i in image_id:
        m = cv2.imread(DATA_DIR +'/test_images/%s'%i)
        image.append(m)
    image = np.stack(image)
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
        
torch version: 1.2.0
input:  torch.Size([4, 3, 256, 1600])
logit:  torch.Size([4, 5, 256, 1600])

[[4.2859354 4.3712254 4.541806  4.673983  4.767757 ]
 [4.237069  4.339175  4.5433893 4.6927257 4.7871857]
 [4.1393356 4.275076  4.5465555 4.7302113 4.826044 ]
 [4.1033278 4.252742  4.551571  4.747407  4.840251 ]
 [4.1290436 4.272175  4.558437  4.744314  4.8298078]] 

[[4.306243  4.1898646 4.0734544 3.9570115 3.89879  ]
 [4.2352004 4.117502  4.000481  3.8841367 3.8259642]
 [4.1956406 4.080703  3.9746747 3.877556  3.8289967]
 [4.187563  4.079467  3.9960358 3.93727   3.9078872]
 [4.183524  4.078849  4.0067163 3.9671268 3.9473324]] 

-0.02721289 3.712041 9.195948 -5.5987377 
 
'''


########################################################################

def run_make_submission_csv():

    threshold_label      = [ 0.75, 0.85, 0.50, 0.50,]
    threshold_mask_pixel = [ 0.40, 0.40, 0.40, 0.40,]
    threshold_mask_size  = [   40,   40,   40,   40,]

    ## load net
    print('load net ...')
    net = Net().cuda()
    net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=lambda storage, loc: storage),strict=True)
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


    ## start here ----------------------------------
    image_id_class_id = []
    encoded_pixel     = []


    net.eval()

    start_timer = timer()
    for t,(input, image_id) in enumerate(loader):
        if t%10==300: #200
            print('\r loader: t = %4d / %4d  %s  %s : %s'%(
                  t, len(loader)-1, str(input.shape), image_id[0], time_to_str((timer() - start_timer),'sec'),
            ),end='', flush=True)

        with torch.no_grad():
            input = input.cuda()

            if 1: # 'null' in augment:
                logit = net(input) #data_parallel(net,input)  #net(input)
                probability = torch.softmax(logit,1)

                probability_mask  = probability[:,1:] #just drop background
                probability_label = probability_mask_to_probability_label(probability)[:,1:]
                num_augment =1

            if 1 : #'flip_lr' in augment:
                logit = net(torch.flip(input,dims=[3]))
                probability  = torch.softmax(torch.flip(logit,dims=[3]),1)

                probability_mask  += probability[:,1:] #just drop background
                probability_label += probability_mask_to_probability_label(probability)[:,1:]
                num_augment +=1

            if 1 : #'flip_ud' in augment:
                logit = net(torch.flip(input,dims=[2]))
                probability = torch.softmax(torch.flip(logit,dims=[2]),1)

                probability_mask  += probability[:,1:] #just drop background
                probability_label += probability_mask_to_probability_label(probability)[:,1:]
                num_augment +=1

            #---
            probability_mask  = probability_mask/num_augment
            probability_label = probability_label/num_augment


        probability_mask  = probability_mask.data.cpu().numpy()
        probability_label = probability_label.data.cpu().numpy()


        batch_size = len(image_id)
        for b in range(batch_size):
            for c in range(4):
                rle=''

                predict_label = probability_label[b,c]>threshold_label[c]
                if predict_label:
                    try:
                        predict_mask = probability_mask[b,c] > threshold_mask_pixel[c]
                        predict_mask = post_process(predict_mask, threshold_mask_size[c])
                        rle = run_length_encode(predict_mask)

                    except:
                        print('An exception occurred : %s'%(image_id[b]+'_%d'%(c+1)))


                image_id_class_id.append(image_id[b]+'_%d'%(c+1))
                encoded_pixel.append(rle)


    print('\r loader: t = %4d / %4d  %s  %s : %s'%(
          t, len(loader)-1, str(input.shape), image_id[0], time_to_str((timer() - start_timer),'sec'),
    ),end='', flush=True)
    print('\n')


    df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv(SUBMISSION_CSV_FILE, index=False)


    ## print statistics ----
    if 1:
        text = summarise_submission_csv(df)
        print(text)

'''

compare with LB probing ... 


    threshold_label      = [ 0.75, 0.85, 0.50, 0.50,]
    threshold_mask_pixel = [ 0.45, 0.45, 0.40, 0.40,]
    threshold_mask_size  = [   40,   40,   40,   40,]
    
		num_image =  1801(1801) 
		num  =  7204(7204) 

		pos1 =    96( 128)  0.750
		pos2 =     7(  43)  0.163
		pos3 =   590( 741)  0.796
		pos4 =   114( 120)  0.950

		neg1 =  1705(1673)  1.019   32
		neg2 =  1794(1758)  1.020   36
		neg3 =  1211(1060)  1.142  151
		neg4 =  1687(1681)  1.004    6
--------------------------------------------------
		neg  =  6397(6172)  1.036  225 
'''

# main #################################################################
if __name__ == '__main__':
    #run_check_setup()
    run_make_submission_csv()

    print('\nsucess!')

