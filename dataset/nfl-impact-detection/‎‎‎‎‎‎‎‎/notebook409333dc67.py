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
print('gpu:', str(torch.cuda.get_device_properties(0))[22:-1])
print('torch verision:', torch.__version__)
print('')



# setting #######################################################
mode = 'kaggle'  # 'kaggle'

if mode == 'local':
    video_dir = '/root/share1/kaggle/2020/nfl/data/nfl-impact-detection' #test
    image_dir = '/root/share1/kaggle/2020/nfl/result/dummy_dump'
    my_data_dir = '/root/share1/kaggle/2020/nfl/code/dummy_01/__submit_kernel__/v00/my_data'
    checkpoint = '/root/share1/kaggle/2020/nfl/result/resnet2d-34-0to200-all-4x-multi-task-2/fold1-v4/checkpoint/00014600_model.pth'

if mode == 'kaggle':
    video_dir = '../input/nfl-impact-detection'
    image_dir = 'dummy_dump'
    my_data_dir = '../input/nfldummy/v00/my_data'
    checkpoint = '../input/nfldummy/00014600_model.pth'

sys.path.insert(0, my_data_dir)
from model import Net


#--------------------------------------------------------------
def do_one_video(net, video_file):
    video = video_file.split('/')[-1]
    gameKey, playID, view = video[:-4].split('_')

    vidcap = cv2.VideoCapture(video_file)
    num_frame = int(cv2.VideoCapture.get(vidcap, int(cv2.CAP_PROP_FRAME_COUNT)))  # 471

    duration = 8
    predict = []
    for frame in range(1, num_frame, duration):
        if frame % duration == 1:
            print('\r frame%03d %s' % (frame, time_to_str(timer() - start_timer, 'min')), end='', flush=True)

        image = []
        for f in range(duration):
            flag, m = vidcap.read()
            if not flag: break
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
        image = image.float().cuda()

        with torch.no_grad():
            net = net.eval()
            prob, box, impact, player = net(image)
            prob = [torch.sigmoid(l).squeeze(1) for l in prob]
            impact = [F.softmax(l, 1) for l in impact]

        # box    = [b.data.cpu().float().numpy().astype(np.int32) for b in box]
        # prob   = [torch.sigmoid(l).squeeze(1).data.cpu().float().numpy() for l in logit]
        # impact = [F.softmax(l, 1).data.cpu().float().numpy() for l in impact]
        # player = [F.softmax(l, 1).data.cpu().float().numpy() for l in player]

        batch_size = len(image)
        for b in range(batch_size):  # batch_size

            predict_box = []
            predict_score = []
            # left , left, top, right, bottom, impact_type
            for s in range(net.num_scale):
                y, x = torch.where(prob[s][b] > 0.5)

                pre_score = prob[s][b][y, x]
                score = 1 - impact[s][b, 0, y, x]
                left, top, right, bottom = box[s][b, :, y, x]
                width = right - left
                height = bottom - top

                predict_box.append(torch.stack([left, width, top, height], 1))
                predict_score.append(score)

            predict_box = torch.cat(predict_box)
            predict_score = torch.cat(predict_score)

            # post process <todo> NMS, top-k
            predict_score = predict_score.data.cpu().numpy()
            predict_box = predict_box.data.cpu().numpy()
            predict_box = np.round(predict_box).astype(np.int32)

            argsort = np.argsort(-predict_score)
            predict_score = predict_score[argsort]
            predict_box = predict_box[argsort]

            keep = np.where(predict_score > 0.5)
            predict_score = predict_score[keep]
            predict_box = predict_box[keep]

            topk = 50
            predict_score = predict_score[:topk]
            predict_box = predict_box[:topk]

            # ---
            for left, width, top, height in predict_box:
                predict.append([gameKey, playID, view, video, frame + b, left, width, top, height])


    print('')
    df_predict = pd.DataFrame(predict, columns=['gameKey', 'playID', 'view', 'video', 'frame', 'left', 'width', 'top',
                                                'height', ])
    return df_predict



# ----
video_file = glob.glob(video_dir + '/test/*.mp4')
# video_file = [video_dir + '/train/' + v for v in[
# '57775_000933_Endzone.mp4', '57775_000933_Sideline.mp4',
# '57906_000718_Endzone.mp4', '57906_000718_Sideline.mp4',
# '57995_000109_Endzone.mp4', '57995_000109_Sideline.mp4',
# '58094_000423_Endzone.mp4', '58094_000423_Sideline.mp4',
# '58094_002819_Endzone.mp4', '58094_002819_Sideline.mp4',
# '58102_002798_Endzone.mp4', '58102_002798_Sideline.mp4']]
print('len(video_file)', len(video_file))
print('')

#---
net = Net().cuda()
net.num_scale = 3
s = net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'], strict=True)
print('net.load_state_dict() ok!', s)


df_submit = []
start_timer = timer()
for j, file in enumerate(video_file):
    # file = '/root/share1/kaggle/2020/nfl/data/nfl-impact-detection/train/57775_000933_Endzone.mp4'
    print(j,file)

    # ------
    df_predict = do_one_video(net, file)
    df_submit.append(df_predict)
    #pd.concat(df_submit).to_csv('submission.csv', index=False)

    print('df_predict.shape', df_predict.shape)
    print('ram:', 'gb available = %d' % int(psutil.virtual_memory().available / 1024 / 1024 / 1024))
    print('')


print('---------------------------------')
df_submit = pd.concat(df_submit).reset_index(drop=True)
print(df_submit)
print(df_submit.shape)
print(df_submit.dtypes)
df_submit.to_csv('submission.csv', index=False)

if mode == 'kaggle':
    import nflimpact
    env = nflimpact.make_env()
    env.predict(df_submit)
    print('env.predict ok')

'''

len(video_file) 6

net.load_state_dict() ok! <All keys matched successfully>
0 /root/share1/kaggle/2020/nfl/data/nfl-impact-detection/test/57906_000718_Sideline.mp4
 frame433  0 hr 00 min
df_predict.shape (22, 9)
ram: gb available = 88

1 /root/share1/kaggle/2020/nfl/data/nfl-impact-detection/test/57906_000718_Endzone.mp4
 frame433  0 hr 01 min
df_predict.shape (42, 9)
ram: gb available = 88

2 /root/share1/kaggle/2020/nfl/data/nfl-impact-detection/test/58102_002798_Sideline.mp4
 frame361  0 hr 01 min
df_predict.shape (39, 9)
ram: gb available = 88

3 /root/share1/kaggle/2020/nfl/data/nfl-impact-detection/test/57995_000109_Endzone.mp4
 frame521  0 hr 02 min
df_predict.shape (9, 9)
ram: gb available = 88

4 /root/share1/kaggle/2020/nfl/data/nfl-impact-detection/test/57995_000109_Sideline.mp4
 frame521  0 hr 03 min
df_predict.shape (63, 9)
ram: gb available = 88

5 /root/share1/kaggle/2020/nfl/data/nfl-impact-detection/test/58102_002798_Endzone.mp4
 frame361  0 hr 03 min
df_predict.shape (88, 9)
ram: gb available = 88

---------------------------------
    gameKey  playID      view  ... width  top  height
0     57906  000718  Sideline  ...    10  365      11
1     57906  000718  Sideline  ...    10  334      11
2     57906  000718  Sideline  ...    10  365      10
3     57906  000718  Sideline  ...    11  365      11
4     57906  000718  Sideline  ...    11  301      12
..      ...     ...       ...  ...   ...  ...     ...
258   58102  002798   Endzone  ...    26  154      33
259   58102  002798   Endzone  ...    27  152      32
260   58102  002798   Endzone  ...    27  151      32
261   58102  002798   Endzone  ...    27  149      22
262   58102  002798   Endzone  ...    27  147      21

[263 rows x 9 columns]
(263, 9)
gameKey    object
playID     object
view       object
video      object
frame       int64
left        int64
width       int64
top         int64
height      int64
dtype: object

Process finished with exit code 0


'''