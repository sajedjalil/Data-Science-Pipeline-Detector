import sys
import os
import cv2
import glob
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import psutil
import platform
import shutil
import gc

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

print('ram:', 'gb=%d' % int(psutil.virtual_memory()[0] / 1024 / 1024 / 1024))
print('cpu:', psutil.cpu_count())
#print('gpu:', str(torch.cuda.get_device_properties(0))[22:-1])
print('torch verision:', torch.__version__)
print('')

start_timer = timer()

# setting #######################################################
mode = 'kaggle'  # 'kaggle'
if mode == 'local':
    video_dir   = '/root/share1/kaggle/2020/nfl/data/nfl-impact-detection' #test
    image_dir   = '/root/share1/kaggle/2020/nfl/result/dummy_dump'
    my_data_dir = '/root/share1/kaggle/2020/nfl/code/dummy_01/__submit_kernel__/v00/my_data'
    checkpoint  = '/root/share1/kaggle/2020/nfl/result/resnet2d-34-0to200-all-4x-multi-task-2/fold1-v4/checkpoint/00014600_model.pth'

if mode == 'kaggle':
    video_dir   = '../input/nfl-impact-detection'
    image_dir   = 'dummy_dump'
    my_data_dir = '../input/nfldummy/v00/my_data'
    checkpoint  = '../input/nfldummy/00014600_model.pth'

sys.path.insert(0, my_data_dir)
 

#--------------------------------------------------------------

# kaggle limit
# https://www.kaggle.com/product-feedback/195163
# disk output : 20 GB


#cache all video to image

# https://www.kaggle.com/artkulak/both-zones-2class-object-detection-strict-filter
# https://stackoverflow.com/questions/26965527/opencv-capturefromcam-memory-leak
def dump_video_file_to_disk(video_file, image_dir):
    os.makedirs(image_dir, exist_ok=True) 
    
    n = 0
    vidcap = cv2.VideoCapture(video_file)
    while True:

        flag, f = vidcap.read()
        if not flag:
            # print('read error at:', video_file)#end of file
            break
        cv2.imwrite(image_dir + '/frame%08d.png' % (n + 1), f)
        n += 1
        
    vidcap.release()
    num_frame = n
    return num_frame



def do_one_video(net, image_dir,  num_frame):

    predict = []
    
    duration = 8
    for frame in range(0, num_frame, duration):
        if frame % duration == 0:
            print('\r frame%03d %s' % (frame, time_to_str(timer() - start_timer, 'min')), end='', flush=True)

        image = []
        for b in range(duration):
            m = cv2.imread(image_dir + '/frame%08d.png' % (frame+b+1))
            if m is None: break
            if m.shape != (720, 1280):
                m = cv2.resize(m, dsize=(1280, 720), interpolation=cv2.INTER_LINEAR)
            image.append(m)
        if image == []:  break
        image = np.stack(image)
        # ---

        image = image.astype(np.float32) / 255
        image = image[..., ::-1]
        image = image.transpose(0, 3, 1, 2)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image)
       
 

        batch_size = len(image)
        for b in range(batch_size):  # batch_size

            predict_box = []
            predict_score = []
             
            topk = 10
            for j in range(topk): 
                left   = np.random.choice(1280-100)
                top    = np.random.choice(720-100)
                width  = np.random.choice(25)+25
                height = np.random.choice(25)+25
                predict_box.append([left, width, top, height])
             
            # ---
            for left, width, top, height in predict_box:
                predict.append([frame+b+1, left, width, top, height])
   
    print('')
    df_predict = pd.DataFrame(predict, columns=['frame', 'left', 'width', 'top', 'height', ])
    return df_predict




###############################################################################

video_file = glob.glob(video_dir + '/test/*.mp4')
print('len(video_file)', len(video_file))
print('dumping video to images ....')

num_frame = {}
def do_dump_video():
    start_timer = timer()
    for j, file in enumerate(video_file):
        # file = '/root/share1/kaggle/2020/nfl/data/nfl-impact-detection/train/57775_000933_Endzone.mp4'
        print(j,file)

        video = file.split('/')[-1]
        dir = image_dir + '/%s'%video[:-4]
        num_frame[video] = dump_video_file_to_disk(file, dir)

        print('tiem: %s' % time_to_str(timer() - start_timer, 'min'))
        print('ram: gb available = %d' % int(psutil.virtual_memory().available / 1024 / 1024 / 1024))
        print('')

 
## start here !!!###################
do_dump_video()
print(str(num_frame).replace(',',',\n'))
print('')


#---
def predict_video():
    net = None
    df_submit = [] 
    for j, file in enumerate(video_file):
        print(j,file)

        # ------
        video = file.split('/')[-1]
        dir = image_dir + '/%s'%video[:-4]
        gameKey, playID, view = video[:-4].split('_')


        df_predict = do_one_video(net, dir, num_frame[video])

        df_predict['gameKey'] = gameKey
        df_predict['playID'] = playID
        df_predict['view']   = view
        df_predict['video']  = video
        df_predict = df_predict[['gameKey', 'playID', 'view', 'video', 'frame', 'left', 'width', 'top','height']]
        df_predict['left'  ] =  df_predict['left'  ].astype(np.int32)
        df_predict['width' ] =  df_predict['width' ].astype(np.int32)
        df_predict['top'   ] =  df_predict['top'   ].astype(np.int32)
        df_predict['height'] =  df_predict['height'].astype(np.int32)

        df_submit.append(df_predict)
        #pd.concat(df_submit).to_csv('submission.csv', index=False)

        print('df_predict.shape', df_predict.shape)
        print('ram:', 'gb available = %d' % int(psutil.virtual_memory().available / 1024 / 1024 / 1024))
        print('')
        
        

    df_submit = pd.concat(df_submit).reset_index(drop=True)
    print(df_submit)
    print(df_submit.shape)
    print(df_submit.dtypes) 
    return df_submit
    
####################################################################
df_submit = predict_video()

os.makedirs(image_dir, exist_ok=True)
shutil.rmtree(image_dir)
df_submit.to_csv('submission.csv', index=False)




print('ok')
#---------------------------------------------------------------------------
import nflimpact
env = nflimpact.make_env()

#df_submit = pd.read_csv('../input/nfl-impact-detection/sample_submission.csv')
env.predict(df_submit)


 