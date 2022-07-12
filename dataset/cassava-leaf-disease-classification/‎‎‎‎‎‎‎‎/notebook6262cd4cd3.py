import sys
import os
import cv2
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import psutil
import platform

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn.functional as F


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError


print('ram:', 'gb=%d'%int(psutil.virtual_memory()[0] / 1024 / 1024 / 1024))
print('cpu:', psutil.cpu_count())
print('gpu:', str(torch.cuda.get_device_properties(0))[22:-1])
print('')

# setting #######################################################


if 0:  # local
    image_dir = '/root/share1/kaggle/2020/cassava/data/train_images'
    df = pd.read_csv('/root/share1/kaggle/2020/cassava/data/train.csv')
    image_id = df.loc[:299, 'image_id'].values.tolist()
    label = df.loc[:299, 'label'].values

    checkpoint0 = [
        '/root/share1/kaggle/2020/cassava/result/effb4/xx4-512/fold0/checkpoint/00011000_model.pth',
        '/root/share1/kaggle/2020/cassava/result/effb4/xx4-512/fold1/checkpoint/00011000_model.pth',
        '/root/share1/kaggle/2020/cassava/result/effb4/xx4-512/fold2/checkpoint/00010000_model.pth',
    ]
    checkpoint1 = [
        '/root/share1/kaggle/2020/cassava/result/se-res50/xx2-512/fold0/checkpoint/00045000_model.pth',
        '/root/share1/kaggle/2020/cassava/result/se-res50/xx2-512/fold2/checkpoint/00012000_model.pth',
    ]

if 1:  # kaggle
    if 0: #(debug)
        image_dir = '../input/cassava-leaf-disease-classification/train_images'
        df = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
        image_id = df.loc[:299, 'image_id'].values.tolist()
        label = df.loc[:299, 'label'].values
    else:
        image_dir = '../input/cassava-leaf-disease-classification/test_images'
        image_id = list(os.listdir(image_dir))
        label = None


    sys.path.insert(0, '../input/cassava-dummy')
    checkpoint0 =  [
#         '../input/cassava-dummy/efb4_fold0_00011000_model.pth',
#         '../input/cassava-dummy/efb4_fold1_00011000_model.pth',
#         '../input/cassava-dummy/efb4_fold2_00010000_model.pth',
#         '../input/cassava-dummy/efb4_fold3_00009000_model.pth',
#         '../input/cassava-dummy/efb4_fold4_00009000_model.pth',
        
        '../input/cassava-dummy/efb4_ext_fold0_00050500_model.pth',
        '../input/cassava-dummy/efb4_ext_fold1_00015000_model.pth',
        '../input/cassava-dummy/efb4_ext_fold2_00014000_model.pth',
        '../input/cassava-dummy/efb4_ext_fold3_00014000_model.pth',
        '../input/cassava-dummy/efb4_ext_fold4_00013500_model.pth',
        
#         '../input/cassava-dummy/effb4_512_new_fold1_00008500_model.pth',
#         '../input/cassava-dummy/effb4_512_new_fold2_00009500_model.pth',
#         '../input/cassava-dummy/effb4_512_new_fold3_00009500_model.pth',
#         '../input/cassava-dummy/effb4_512_new_fold4_00009000_model.pth',
         
        
    ]
    checkpoint1 =  [
#         '../input/cassava-dummy/se50_fold0_00045000_model.pth',
#         '../input/cassava-dummy/se50_fold1_00012000_model.pth',
#         '../input/cassava-dummy/se50_fold2_00012000_model.pth',
#         '../input/cassava-dummy/se50_fold3_00011000_model.pth',
#         '../input/cassava-dummy/se50_fold4_00011000_model.pth',
        
        
#         '../input/cassava-dummy/se50_ext_fold0_00050500_model.pth',
#         '../input/cassava-dummy/se50_ext_fold1_00015000_model.pth',
#         '../input/cassava-dummy/se50_ext_fold2_00016000_model.pth',
#         '../input/cassava-dummy/se50_ext_fold3_00017500_model.pth',
#         '../input/cassava-dummy/se50_ext_fold4_00015500_model.pth', 
        
        
#         '../input/cassava-dummy/se50_448_fold0_00014000_model.pth',
#         '../input/cassava-dummy/se50_448_fold1_00015000_model.pth',
#         '../input/cassava-dummy/se50_448_fold2_00015000_model.pth',
#         '../input/cassava-dummy/se50_448_fold3_00014000_model.pth',
#         '../input/cassava-dummy/se50_448_fold4_00014000_model.pth',
    ]


#################################################################
from model0 import Net as Net0  #efficient-b4
from model1 import Net as Net1  #se-reesnext-50


# model ---
net0 = []
for f in checkpoint0:
    n = Net0().cuda()
    n.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage)['state_dict'] , strict=True)
    net0.append(n)

net1 = []
for f in checkpoint1:
    n = Net1().cuda()
    n.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage)['state_dict'], strict=True)
    net1.append(n)

 
print('load checkppoint ok! net0', len(net0))
print('load checkppoint ok! net1', len(net1))


# dataset ---
image_size = 512

class CassavaDataset(Dataset):
    def __init__(self, ):
        pass

    def __len__(self):
        return len(image_id)

    def __getitem__(self, index):
        image = cv2.imread(image_dir + '/%s' % image_id[index])
        image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
        image = image[...,::-1]
        image = image.astype(np.float32) / 255
        image = np.ascontiguousarray(image.transpose(2, 1, 0))
        return image


dataset = CassavaDataset()
loader = DataLoader(
    dataset,
    sampler=SequentialSampler(dataset),
    batch_size=16,
    drop_last=False,
    num_workers=0,
    pin_memory=True,
)

# start here! ------------------
probability = []

start_timer = timer()
with torch.no_grad():

    for t, batch in enumerate(loader):
        image = batch.cuda()

        p = []
        for net in net0 + net1: #
            net.eval()
            logit = net(image)
            p.append(F.softmax(logit, -1))

            # tta ----
            if 0:
                logit = net(torch.flip(image, dims=(2,)).contiguous())
                p.append(F.softmax(logit, -1))

                logit = net(torch.flip(image, dims=(3,)).contiguous())
                p.append(F.softmax(logit, -1))

                logit = net(torch.flip(image, dims=(2,3)).contiguous())
                p.append(F.softmax(logit, -1))

                logit = net(image.permute(0,1,3,2).contiguous())
                p.append(F.softmax(logit, -1))

        # ---------
        p = torch.stack(p).mean(0)
        probability.append(p.data.cpu().numpy())

        if t % 3 == 0:
            print('\r %8d / %d  %s' % (t, len(loader), time_to_str(timer() - start_timer, 'sec')), end='', flush=True)
    print('')

# ----------------------
probability = np.concatenate(probability)
predict = probability.argmax(-1)

df_submit = pd.DataFrame({'image_id': image_id, 'label': predict})
print('df_submit', df_submit.shape)
print(df_submit)
print('estimated time for 15,000 test images = %s'%time_to_str((timer() - start_timer)/len(predict)*15000, 'min'))

df_submit.to_csv('submission.csv', index=False)

# check
if label is not None:
    correct = (predict == label).mean()
    print('correct', correct)
    print('probability\n', probability[:5])
    print('predict\n', predict[:10])

'''
load checkppoint ok!
       18 / 19   2 min 39 sec
df_submit (300, 2)
           image_id  label
0    1000015157.jpg      2
1    1000201771.jpg      3
2     100042118.jpg      4
3    1000723321.jpg      1
4    1000812911.jpg      3
..              ...    ...
295  1052095724.jpg      0
296  1052118637.jpg      3
297  1052854295.jpg      0
298  1052881053.jpg      3
299  1052903541.jpg      3

[300 rows x 2 columns]
estimated time for 15,000 test images =  2 hr 12 min
correct 0.93
probability
 [[3.09394360e-01 1.13561094e-01 3.63957703e-01 3.47374426e-03
  2.09613070e-01]
 [1.88316408e-05 1.86062753e-04 1.49351486e-03 9.98232782e-01
  6.88106447e-05]
 [1.04819878e-03 2.20503122e-01 6.97037764e-03 1.38261551e-02
  7.57652104e-01]
 [1.67905848e-04 9.92914855e-01 3.28231661e-04 3.20969219e-03
  3.37923691e-03]
 [1.42867793e-05 1.11910940e-05 3.97972763e-04 9.99195218e-01
  3.81269114e-04]]
predict
 [2 3 4 1 3 3 2 0 0 3]

Process finished with exit code 0

'''