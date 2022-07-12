from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import cv2

def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]

def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR) #use cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return [path, resized]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata

train = glob.glob('../input/train/**/*.jpg') + glob.glob('../input/additional/**/*.jpg')
#train=glob.glob('../input/train/Type_1/*.jpg')[:5] +glob.glob('../input/train/Type_2/*.jpg')[:5] +glob.glob('../input/train/Type_3/*.jpg')[:5]
train = pd.DataFrame([[p.split('/')[3],p.split('/')[4],p] for p in train], columns = ['type','image','path']) #limit for Kaggle Demo
train = im_stats(train)
train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
train_data = normalize_image_features(train['path'])
np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)

le = LabelEncoder()
train_target = le.fit_transform(train['type'].values)
print(le.classes_) #in case not 1 to 3 order
np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)