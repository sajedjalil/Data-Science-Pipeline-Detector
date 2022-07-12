#!/usr/bin/env python
# coding: utf-8

# In[0]:

import pandas as pd
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
import sys
import gc

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.nn import functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.models as models

import matplotlib.patches as patches
import random
import os
import glob
from pandas.io.json import json_normalize
from joblib import Parallel, delayed
import time
import tqdm.notebook as tqdm
import pdb
import subprocess
import xxhash
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from base64 import b64encode
from IPython.display import HTML
import platform
import shutil
from scipy.io import wavfile
import hashlib
from sklearn.metrics import mean_squared_error,log_loss,roc_auc_score
from ast import literal_eval
from matplotlib.animation import FuncAnimation, ArtistAnimation
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import albumentations as A
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from collections import namedtuple
from torchvision.ops import nms as torch_nms
from collections import OrderedDict

import albumentations.augmentations.functional as AF


CLOUD = (not torch.cuda.is_available()) or \
            ((torch.cuda.get_device_name(0) != 'GeForce GTX 960M') & (torch.cuda.get_device_name(0) !='Tesla V100-PCIE-32GB'))
#KAGGLE = (platform.platform() == 'Linux-4.9.0-11-amd64-x86_64-with-debian-9.9')
KAGGLE = Path('/kaggle/working').is_dir()
TPU = 'TPU_IP_ADDRESS' in list(os.environ)
YUVAL = (torch.cuda.is_available()) and (torch.cuda.get_device_name(0) =='Tesla V100-PCIE-32GB')
YUVAL_DEVICE=0
MAX_DEVICES = 8
KAGGLE_TEST = False
print (YUVAL)
MTCNN_IMAGES_PER_VIDEO = 16
IMAGES_PER_VIDEO = 32
SAMPLES_PER_AUDIO = 7
SHORT_RUN=False

if False:
    KAGGLE_TEST = True
    KAGGLE = True
    SHORT_RUN=False

if CLOUD and KAGGLE and (not KAGGLE_TEST):
    PATH = Path('/kaggle/input/deepfake-detection-challenge/')
    PATH_DISK = PATH
    PATH_FFMPEG = Path('/kaggle/working/ffmpeg-git-20191209-amd64-static/')
    PATH_WORK = Path('/kaggle/working/')
    PATH_MODELS = Path('/kaggle/input/dfdc-models5')
    PATH_META = Path('/kaggle/input/train-set-metadata-for-dfdc')
    PATH_DSFD_WEIGHTS = Path('/kaggle/input/facenetpytorch/WIDERFace_DSFD_RES152.pth')
    PATH_LIGHT_DSFD_WEIGHTS = Path('/kaggle/input/facenetpytorch/dsfdv2_r18.pth')
elif CLOUD and (not KAGGLE) and (not KAGGLE_TEST):
    PATH = Path('/home/zahar_chikishev/DFDC')
    PATH_DISK = Path('/mnt/disk')
    PATH_FFMPEG = PATH/'ffmpeg-4.2.2-amd64-static'
    PATH_WORK = PATH_DISK
    PATH_MODELS = PATH_DISK/'models'
    PATH_META = PATH
    PATH_DSFD_WEIGHTS = PATH_DISK/'dsfd_weights/WIDERFace_DSFD_RES152.pth'
    PATH_LIGHT_DSFD_WEIGHTS = PATH_DISK/'light_dsfd_weights/dsfdv2_r18.pth'
elif CLOUD and KAGGLE_TEST:
    PATH = Path('/home/zahar_chikishev/DFDC')
    PATH_DISK = Path('/mnt/disk')
    PATH_FFMPEG = PATH/'ffmpeg-4.2.2-amd64-static'
    PATH_WORK = PATH_DISK/'temp'
    PATH_MODELS = PATH_DISK/'models'
    PATH_META = PATH
    PATH_DSFD_WEIGHTS = PATH_DISK/'dsfd_weights/WIDERFace_DSFD_RES152.pth'
    PATH_LIGHT_DSFD_WEIGHTS = PATH_DISK/'light_dsfd_weights/dsfdv2_r18.pth'
elif YUVAL:
    PATH = Path('/workspace/nvme')
    PATH_DISK = PATH #Path('/workspace/hd')
    PATH_FFMPEG = PATH/'ffmpeg-git-20191209-amd64-static/'
    PATH_WORK = Path('/workspace/hd')
    PATH_MODELS = Path('/workspace/hd')
    PATH_DSFD_WEIGHTS = Path('/workspace/notebooks/deepfake/zahar/DFDC/dsfd_weights/WIDERFace_DSFD_RES152.pth')
    PATH_LIGHT_DSFD_WEIGHTS = Path('/workspace/notebooks/deepfake/zahar/DFDC/light_dsfd_weights/dsfdv2_r18.pth')

    PATH_META = PATH    
elif KAGGLE_TEST:
    PATH = Path('C:\\StudioProjects\\DFDC')
    PATH_DISK = Path('D:\\DFDC')
    PATH_FFMPEG = PATH/'ffmpeg-20191219-99f505d-win64-static/bin'
    PATH_WORK = Path('D:\\DFDC\\temp')
    PATH_MODELS = Path('D:\\DFDC\\dataset')
    PATH_META = PATH
    PATH_DSFD_WEIGHTS = Path('C:\\StudioProjects\\DFDC\\dsfd_weights\\WIDERFace_DSFD_RES152.pth')
    PATH_LIGHT_DSFD_WEIGHTS = Path('C:\\StudioProjects\\DFDC\\light_dsfd_weights\\dsfdv2_r18.pth')
else:
    PATH = Path('C:\StudioProjects\DFDC')
    PATH_DISK = Path('D:\DFDC')
    PATH_FFMPEG = PATH/'ffmpeg-20191219-99f505d-win64-static/bin'
    PATH_WORK = PATH_DISK
    PATH_MODELS = PATH_DISK/'models'
    PATH_META = PATH
    PATH_DSFD_WEIGHTS = Path('C:\\StudioProjects\\DFDC\\dsfd_weights\\WIDERFace_DSFD_RES152.pth')
    PATH_LIGHT_DSFD_WEIGHTS = Path('C:\\StudioProjects\\DFDC\\light_dsfd_weights\\dsfdv2_r18.pth')

if (not KAGGLE) or KAGGLE_TEST:
    from dlib import get_frontal_face_detector
    import dlib
    if torch.cuda.is_available():
        import face_recognition
    from facenet_pytorch_local import MTCNN, InceptionResnetV1, extract_face
    import jupyter_contrib_nbextensions
    import dsfd_pytorch_local.dsfd as dsfd
    
    import skvideo
    skvideo.setFFmpegPath(str(PATH_FFMPEG))
    import skvideo.io
    
    import dill
    if not YUVAL:
        sys.path.insert(0, str(PATH/"light_dsfd_local/DSFDv2_r18"))
    else:
        sys.path.insert(0, "/workspace/notebooks/deepfake/zahar/DFDC/light_dsfd_local/DSFDv2_r18")
    from light_dsfd_local.DSFDv2_r18.model_search import Network



if CLOUD and TPU:
    import torch_xla
    import torch_xla.distributed.data_parallel as dp
    import torch_xla.utils.utils as xu
    import torch_xla.core.xla_model as xm
    
if YUVAL:
    import pretrainedmodels

if CLOUD or YUVAL:
    FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
else:
    FONT_PATH = 'arial.ttf'
if YUVAL:
    device = 'cuda:{}'.format(YUVAL_DEVICE)
    torch.cuda.set_device(device)
else:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if TPU:
    device = xm.xla_device()

#device = 'cpu'
print(f'Running on device: {device}')

if CLOUD:
    splitter = '/'
    if KAGGLE:
        sudo = ''
    else:
        sudo = 'sudo '
elif YUVAL:
    sudo = ''
    splitter = '/'
else:
    splitter = '\\'
    sudo = ''

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

if KAGGLE and (not KAGGLE_TEST):
    ds1_bs = 4
    ds0_bs = 32
else:
    ds1_bs = 1
    ds0_bs = 32

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(2020)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# display current status
#from fastai.utils.show_install import show_install; show_install()


# In[59]:


if TPU:
    devices = xm.get_xla_supported_devices(max_devices=MAX_DEVICES)


# In[60]:


VERSION = 59
DATA_SMALL = False
CLOUD_SINGLE = True
WEIGHTED = False
LOADER_SCALE = 0.2
if KAGGLE_TEST:
    NUM_WORKERS = 0
elif KAGGLE:
    NUM_WORKERS = 2
elif CLOUD:
    NUM_WORKERS = 4
else:
    NUM_WORKERS = 0
bs = 16
learning_rate = 1e-4
weight_decay = 0
NUM_FAKE_CLASSES = 256
NUM_PARTITIONS = 5
MAX_FRAME = 350
BOX_SIZE = (145,185)
FACES_FOLDER = 'faces'
#FACES_FOLDER = 'faces_jpg_from_png'

if KAGGLE:
    dsfd_bs = 8
    fp_bs = 16
elif CLOUD:
    dsfd_bs = 32
    fp_bs = 32
else:
    dsfd_bs = 4
    fp_bs = 8

meta_cols = ['filename', 'label', 'weight', 'target', 'valid', 'fold5', 'prob0_mean_scaled', 'res',
             'anum', 'type', 'original','selected']
drop_cols = ['pxl_mean']
FOLDS_VALID = 2

# In[61]:


pixel_cols = ['pxl_mean', 'sp0', 'sp1', 'sp2', 'sp3', 'sp4']
sample_pixels = [[0.2,0.2], [0.8,0.2], [0.2,0.8], [0.8,0.8], [0.5,0.5]]
err_filenames = [[] for i in range(2)]


# In[62]:


if KAGGLE:
    DATA_SMALL = False
    CLOUD_SINGLE = True


run_face_recognition = False
run_facenet_pytorch = True
run_cascade_classifier = False
run_dlib = False
run_dsfd = False
run_abhishek=False
run_yuval = True
run_zahar = True
# # Utility scripts

# In[63]:


def get_filepaths(mode=None, filtering=True):
    
    if mode is None:
        if not CLOUD or KAGGLE:
            mode = 'sample'
        else:
            mode = 'train'
    
    if mode == 'sample':
        filepaths = glob.glob(str(PATH_DISK/'train_sample_videos'/'*.mp4'))
    elif mode == 'train':
        filepaths = glob.glob(str(PATH_DISK/'dfdc_train_part_*/*.mp4'))
    elif mode == 'ffpp':
        filepaths = glob.glob(str(PATH_DISK/'ffpp/*/*.mp4'))
    elif mode == 'dessa':
        filepaths = glob.glob(str(PATH_DISK/'dessa/*/*.mp4'))
    elif mode == 'crop_benchmark':
        filepaths = glob.glob(str(PATH_DISK/'crop_benchmark/mp4/*.mp4'))
    elif mode == 'special_test':
        filepaths = glob.glob(str(PATH_DISK/'special_test/*.mp4'))
    elif mode == 'valid':
        filepaths = glob.glob(str(PATH_DISK/'test_videos'/'*.mp4'))
    elif mode in ['test_faceforensics', 'test_faceforensics2']:
        filepaths = glob.glob(str(PATH_DISK/mode/'*/*/c40/videos/*.mp4'))
    else:
        assert False
    
    if (mode != 'valid') and filtering:
        duplicates = np.load(PATH/'duplicates.npy', allow_pickle=True)
        filepaths = [filepaths[i] for i,f in enumerate(path2name(filepaths)) if f not in duplicates]
    
    return filepaths


# In[64]:


def enrich_filepath(filenames, mode):
    if mode == 'test_faceforensics':
        return filenames
    filepaths = get_filepaths(mode, filtering=False)
    fns = path2name(filepaths)
    df = pd.DataFrame({'filename':fns, 'filepath':filepaths})
    ret = df.set_index('filename', drop=True).sort_index().loc[filenames]
    return ret.filepath.values


# In[65]:


def assign_boxes(boxes, probs):
    
    N = len(boxes)
    boxes = [[] if bb is None else bb for bb in boxes]
    centers = [[np.array([0.5*(box[0]+box[2]), 0.5*(box[1]+box[3])]) for box in bb] for bb in boxes]
    index = [np.zeros(len(c), dtype=int) for c in centers]
    dist = np.array([np.linalg.norm(c1-c2) for k in range(N-1) 
            for i1,c1 in enumerate(centers[k]) for i2,c2 in enumerate(centers[k+1])])
    point = [(k,i1,i2) for k in range(N-1) for i1 in range(len(centers[k])) for i2 in range(len(centers[k+1]))] 
    
    while True:
        
        if len(dist) == 0:
            break

        min_dist = np.min(dist)
        min_idx = np.argmin(dist)

        if dist[min_idx] > 150: break

        k,i1,i2 = point[min_idx]

        max_face_index = np.concatenate(index).max()

        face_index = max(index[k][i1], index[k+1][i2])
        if face_index == 0: face_index = max_face_index + 1
        if index[k][i1] == 0: index[k][i1] = face_index
        if index[k+1][i2] == 0: index[k+1][i2] = face_index
        index_to_change = None
        if index[k][i1] != face_index: index_to_change = index[k][i1]
        if index[k+1][i2] != face_index: index_to_change = index[k+1][i2]
        if index_to_change is not None:
            index = [np.where(arr==index_to_change,face_index,arr) for arr in index]
        
        closing_links = [((pk == k) & ((i1==pi1) | (i2==pi2))) for pk in range(N-1) 
                         for pi1 in range(len(centers[pk])) for pi2 in range(len(centers[pk+1]))]
        dist[closing_links] = 1000
    
    cur = 1000
    for i in range(N):
        for k in range(len(index[i])):
            if index[i][k] == 0:
                index[i][k] = cur
                cur += 1
    
    indices = np.concatenate(index)
    all_index = np.unique(indices)
    all_probs = np.concatenate(probs)
    all_probs = np.array([p for p in all_probs if p is not None])
    all_weights = [all_probs[indices == i].sum() for i in all_index]
    order = np.argsort(all_weights)[::-1]
    
    mapped_index = np.zeros(len(all_index), dtype=int)
    mask = np.zeros((2,N), dtype=bool)
    for o in order:
        index_mask = np.array([all_index[o] in idx for idx in index])
        sel = 2
        for i in range(2):
            if mask[i, index_mask].max() == 0:
                sel = i
                mask[i] += index_mask
                break
        mapped_index[o] = sel
    
    new_index = index.copy()
    for i in range(len(all_index)):
        new_index = [np.where(arr==all_index[i],mapped_index[i],new_index[k]) for k,arr in enumerate(index)]
    
    return new_index


# In[66]:


def filter_boxes(face_index, probs, boxes, points, selected, selected_mtcnn):
    
    N = len(boxes)
    
    indices = np.concatenate(face_index)
    all_probs = np.concatenate(probs)
    all_probs = np.array([p for p in all_probs if p is not None])
    weight0 = all_probs[indices == 0].sum()
    weight1 = all_probs[indices == 1].sum()

    def interpolate_boxes(box):
        
        len0 = np.array([len(bb) for bb in box])
        N = len(box)
        assert np.all(len0 <= 1)
        
        if len0[0] == 0:
            len_miss = np.where(np.cumsum(len0) > 0)[0][0]
            for i in range(len_miss):
                box[i] = box[len_miss]
            len0 = np.array([len(bb) for bb in box])
        if len0[-1] == 0:
            len_miss = np.where(np.cumsum(len0[::-1]) > 0)[0][0]
            for i in range(len_miss):
                box[N-i-1] = box[N-len_miss-1]
            len0 = np.array([len(bb) for bb in box])
        
        box_shrink = [bb[0] for bb in box if (len(bb) > 0)]

        box_dim = len(box_shrink[0])

        # kind='cubic'
        f_inter = [interp1d(selected_mtcnn[len0 == 1],
                            [bb[k] for bb in box_shrink], kind='linear') for k in range(box_dim)]

        box = np.array([f_inter[k](selected) for k in range(box_dim)]).transpose()
        box = list(np.expand_dims(box,1))
        
        return box

    boxes0 = None
    boxes1 = None
    pnts0 = None
    pnts1 = None
    prb0 = None
    prb1 = None

    if weight0 > 0.5*N:
        boxes0  = [[] if bb is None else bb[idx == 0] for idx,bb in zip(face_index,boxes)]
        boxes0 = interpolate_boxes(boxes0)
        if len(points) > 0:
            pnts0  = [[] if pp is None else pp[idx == 0] for idx,pp in zip(face_index,points)]
            pnts0 = [pp.reshape((len(pp),-1)) if len(pp) > 0 else pp for pp in pnts0]
            pnts0 = interpolate_boxes(pnts0)
            pnts0 = [pp.reshape((len(pp),-1,2)) for pp in pnts0]
        prb0 = [[] if p is None else np.array([[p1] for p1 in p])[idx == 0] for idx,p in zip(face_index,probs)]
        prb0 = interpolate_boxes(prb0)
    if weight1 > 0.5*N:
        boxes1  = [[] if bb is None else bb[idx == 1] for idx,bb in zip(face_index,boxes)]
        boxes1 = interpolate_boxes(boxes1)
        if len(points) > 0:
            pnts1  = [[] if pp is None else pp[idx == 1] for idx,pp in zip(face_index,points)]
            pnts1 = [pp.reshape((len(pp),-1)) if len(pp) > 0 else pp for pp in pnts1]
            pnts1 = interpolate_boxes(pnts1)
            pnts1 = [pp.reshape((len(pp),-1,2)) for pp in pnts1]
        prb1 = [[] if p is None else np.array([[p1] for p1 in p])[idx == 1] for idx,p in zip(face_index,probs)]
        prb1 = interpolate_boxes(prb1)
    
    return boxes0, boxes1, pnts0, pnts1, prb0, prb1


# In[67]:


def margin_boxes(boxes):
    if boxes is None:
        return None
    margin = 14
    m_box = np.array([[-margin, -margin, margin, margin]])
    boxes = [(bb + m_box) for bb in boxes]
    return boxes


# In[68]:


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# In[69]:

def ld_test_base_transform(image, mean):
    #x = cv2.resize(image, (size, size)).astype(np.float32)
    x = image.astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class LDTestBaseTransform:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return ld_test_base_transform(image, self.mean), boxes, labels

light_dsfd_transform = LDTestBaseTransform((104, 117, 123))

def light_dsfd_preprocess(img):
    x = torch.from_numpy(light_dsfd_transform(img)[0]).permute(2, 0, 1)
    x = x.unsqueeze(0).cuda()
    return x


def get_detection_model(ds_fc=False, fp_fc=False, dl_fc=False, bl_fc=False, ld_fc=False):
    
    if fp_fc:
        
        if ('mtcnn' not in locals()) and ('mtcnn' not in globals()):
            global mtcnn
            mtcnn = MTCNN(device=device, keep_all=True, select_largest=False, min_face_size=60, 
                          factor=0.6, thresholds=[0.6, 0.7, 0.7]).eval()
            print("MTCNN model loaded")
        
        return mtcnn
    
    elif ds_fc:
        
        if ('dsfd_detector' not in locals()) and ('dsfd_detector' not in globals()):
            global dsfd_detector
            dsfd_detector = dsfd.detect.DSFDDetector(PATH_DSFD_WEIGHTS)
            print("DSFD model loaded")
        
        return dsfd_detector

    elif dl_fc:

        if ('dlib_detector' not in locals()) and ('dlib_detector' not in globals()):
            global dlib_detector
            dlib_detector = get_frontal_face_detector()
            print('dlib detector loaded')
        
        return dlib_detector

    elif ld_fc:

        if ('light_dsfd_detector' not in locals()) and ('light_dsfd_detector' not in globals()):
            global light_dsfd_detector

            FPN_Genotype = namedtuple("FPN_Genotype", "Inter_Layer Out_Layer")
            AutoFPN = FPN_Genotype(
                Inter_Layer=[
                    [("sep_conv_3x3", 1), ("conv_1x1", 0)],
                    [("sep_conv_3x3", 2), ("sep_conv_3x3", 0), ("conv_1x1", 1)],
                    [("sep_conv_3x3", 3), ("sep_conv_3x3", 1), ("conv_1x1", 2)],
                    [("sep_conv_3x3", 4), ("sep_conv_3x3", 2), ("conv_1x1", 3)],
                    [("sep_conv_3x3", 5), ("sep_conv_3x3", 3), ("conv_1x1", 4)],
                    [("sep_conv_3x3", 4), ("conv_1x1", 5)],
                ],
                Out_Layer=[],
            )
            light_dsfd_detector = Network(
                C=64,
                criterion=None,
                num_classes=2,
                layers=1,
                phase="test",
                search=False,
                args={'cuda':True},
                searched_fpn_genotype=AutoFPN,
                searched_cpm_genotype=None,
                fpn_layers=1,
                cpm_layers=1,
                auxiliary_loss=False,
            )

            state_dict = torch.load(PATH_LIGHT_DSFD_WEIGHTS)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "auxiliary" not in k:
                    name = k[7:]
                    new_state_dict[name] = v
                else:
                    print("Auxiliary loss is used when retraining.")

            light_dsfd_detector.load_state_dict(new_state_dict)
            light_dsfd_detector.cuda()
            light_dsfd_detector.eval()

            print('light dsfd detector loaded')
        
        return light_dsfd_detector

    elif bl_fc:

        if ('blaze_net' not in locals()) and ('blaze_net' not in globals()):
            from blazeface_local import blazeface

            global blaze_net
            blaze_net = blazeface.BlazeFace().to(torch.device("cuda:0"))
            blaze_net.load_weights("C:\\StudioProjects\\DFDC\\blazeface_local\\blazeface.pth")
            blaze_net.load_anchors("C:\\StudioProjects\\DFDC\\blazeface_local\\anchors.npy")

            # Optionally change the thresholds:
            blaze_net.min_score_thresh = 0.75
            blaze_net.min_suppression_threshold = 0.3
            print('blaze_net detector loaded')
        
        return blaze_net


# In[70]:


def del_detection_models():
    if ('mtcnn' in locals()) or ('mtcnn' in globals()):
        global mtcnn
        del mtcnn
        print("MTCNN model deleted")
    if ('dsfd_detector' in locals()) or ('dsfd_detector' in globals()):
        global dsfd_detector
        del dsfd_detector
        print("DSFD model deleted")
    if ('light_dsfd_detector' in locals()) or ('light_dsfd_detector' in globals()):
        global light_dsfd_detector
        del light_dsfd_detector
        print("DSFD model deleted")


# In[71]:


def mtcnn_scale_images(images_mtcnn):
    
    shrink = 900/np.array(images_mtcnn[0].shape)[:2].mean()
    if (shrink >= 0.8) and (shrink <= 1.2):
        shrink = 1
    else:
        width = int(images_mtcnn[0].shape[1] * shrink)
        height = int(images_mtcnn[0].shape[0] * shrink)
        dim = (width, height)
        
        images_mtcnn = [cv2.resize(image, dim, interpolation = cv2.INTER_AREA) for image in images_mtcnn]
    
    return images_mtcnn, shrink


# In[72]:


def face_detection_batch(images_mtcnn, ds_fc = False, fp_fc = False, fr_fc = False, dl_fc = False, bl_fc = False, ld_fc = False):
    
    boxes = []; probs = []; points = []
    
    if fp_fc:
        
        detector = get_detection_model(fp_fc = True)
        
        st = time.time()
        images_mtcnn, shrink = mtcnn_scale_images(images_mtcnn)
        
        with torch.no_grad():
#             images_min = np.array([b.min() for b in images_mtcnn]).min()
#             images_max = np.array([b.max() for b in images_mtcnn]).max()
#             images_mtcnn = [((image - images_min)/(images_max - images_min)).astype('uint8') for image in images_mtcnn]
            for batch_images in chunks(images_mtcnn, fp_bs):
                boxes2, probs2, points2 = detector.detect(batch_images, landmarks=True)
                boxes.append([bb/shrink if bb is not None else None for bb in boxes2])
                probs.append(probs2)
                points.append([pp/shrink if pp is not None else None for pp in points2])

                del boxes2, probs2, points2
            
            boxes = [f for sublist in boxes for f in sublist]
            probs = [f for sublist in probs for f in sublist]
            points = [f for sublist in points for f in sublist]
        
        st = time.time() - st
        
    elif ds_fc:
        
        detector = get_detection_model(ds_fc = True)
        
        st = time.time()
        
        with torch.no_grad():
            
            shrink = 300/np.array(images_mtcnn[0].shape)[:2].mean()
            
            for batch_images in chunks(images_mtcnn, dsfd_bs):
                
                detections = detector.detect_face(batch_images, confidence_threshold=.4, shrink=shrink)
                
                for d in detections:
                    boxes.append(d[:,0:4])
                    probs.append(d[:,4])

                del detections
        
        st = time.time() - st
        
    elif fr_fc:
        
        st = time.time()
        images_mtcnn, shrink = mtcnn_scale_images(images_mtcnn)
        
        for image in images_mtcnn:
            with torch.no_grad():
                face_positions = face_recognition.face_locations(image)

            boxes2 = []
            probs2 = []
            margin = 0

            for fp in face_positions:
                offset = round(margin * (fp[2] - fp[0]))
                y0 = max(fp[0] - offset, 0)
                x1 = min(fp[1] + offset, image.shape[1])
                y1 = min(fp[2] + offset, image.shape[0])
                x0 = max(fp[3] - offset, 0)
                boxes2.append(np.array([x0,y0,x1,y1])/shrink)
                probs2.append(1)

            boxes.append(np.array(boxes2))
            probs.append(probs2)

        st = time.time() - st
        
    elif dl_fc:
        
        detector = get_detection_model(dl_fc = True)

        st = time.time()
        images_mtcnn, shrink = mtcnn_scale_images(images_mtcnn)

        for image in images_mtcnn:
            detections = detector(np.array(image))
            probs2 = []
            boxes2 = []

            for det in detections:
                boxes2.append(np.array([det.left(),det.top(),det.right(),det.bottom()])/shrink)
                probs2.append(1)

            boxes.append(np.array(boxes2))
            probs.append(probs2)
            
        st = time.time() - st

    elif bl_fc:

        detector = get_detection_model(bl_fc = True)

        st = time.time()

        width =  images_mtcnn[0].shape[1]
        height = images_mtcnn[0].shape[0]
        
        images_mtcnn = [cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA) for image in images_mtcnn]
        detections = detector.predict_on_batch(np.stack(images_mtcnn))

        #pdb.set_trace()

        for dd in detections:
            probs2 = []
            boxes2 = []

            for d in dd.cpu().numpy():
                boxes2.append(np.array([width*d[1],height*d[0],width*d[3],height*d[2]]))
                probs2.append(d[-1])
            
            boxes.append(np.array(boxes2))
            probs.append(probs2)

        st = time.time() - st

    elif ld_fc:

        detector = get_detection_model(ld_fc = True)

        st = time.time()
        sz = images_mtcnn[0].shape
        scale = torch.Tensor(
            [
                sz[1],
                sz[0],
                sz[1],
                sz[0]
            ]
        )
        
        images_mtcnn, shrink = mtcnn_scale_images(images_mtcnn)

        for image in images_mtcnn:

            imt = light_dsfd_preprocess(np.array(image))

            detections = detector(imt).view(-1, 5)
            detections = detections.detach()

            probs2 = []
            boxes2 = []

            scores = detections[..., 0]
            boxes_all = detections[..., 1:] * scale

            keep_mask = (scores >= 0.7) & (boxes_all[..., -1] > 2.0)
            scores = scores[keep_mask]
            boxes_all = boxes_all[keep_mask]

            keep_idx = torch_nms(boxes_all, scores, iou_threshold=0.4)
            keep_boxes = np.array([])
            keep_scores = np.array([])

            if len(keep_idx) > 0:
                keep_boxes = boxes_all[keep_idx].cpu().numpy()
                keep_scores = scores[keep_idx].cpu().numpy()

            boxes.append(keep_boxes)
            probs.append(keep_scores)
        
        st = time.time() - st

    else:
        assert False
    
    return boxes, probs, points, st


# In[73]:


def run_face_detection(images_mtcnn, combo_fc, selected, selected_mtcnn, **kwargs):
    
    st_total = 0
    
    if combo_fc:
        for method in ['fp_fc','ld_fc']:
            #print('applying method', method)
            kwargs[method] = True
            boxes, probs, points, st = face_detection_batch(images_mtcnn, **kwargs)
            st_total += st
            kwargs[method] = False

            face_index = assign_boxes(boxes, probs)
            boxes0, boxes1, pnts0, pnts1, prb0, prb1 = filter_boxes(face_index, probs, boxes, points, selected, selected_mtcnn)

            if boxes0 is not None:
                break

    else:
        boxes, probs, points, st_total = face_detection_batch(images_mtcnn, **kwargs)

        face_index = assign_boxes(boxes, probs)
        boxes0, boxes1, pnts0, pnts1, prb0, prb1 = filter_boxes(face_index, probs, boxes, points, selected, selected_mtcnn)

    boxes0 = margin_boxes(boxes0)
    boxes1 = margin_boxes(boxes1)
    
    return boxes0, boxes1, pnts0, pnts1, prb0, prb1, st_total


# In[74]:


def get_images(filepath, n=IMAGES_PER_VIDEO, process=True, return_boxes=False, combo_fc=False, **kwargs):
    
    cap = cv2.VideoCapture(str(filepath))
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    selected = np.linspace(0,v_len-1,n-4,dtype=int)
    mid_num = selected[int(n/2)]
    selected = np.sort(np.concatenate([selected, np.array([mid_num-2,mid_num-1,mid_num+1,mid_num+2])]))
    selected_mtcnn = np.linspace(0,v_len-1,MTCNN_IMAGES_PER_VIDEO,dtype=int)

    images_mtcnn = []
    images_orig = []
    for j in range(v_len):
        success = cap.grab()
        if (j in selected) or (j in selected_mtcnn):
            success, vframe = cap.retrieve()
            image = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            
            if j in selected:
                images_orig.append(image)
            
            if j in selected_mtcnn:
                images_mtcnn.append(image)
    
    cap.release()
    
    if process:

        boxes0, boxes1, pnts0, pnts1, prb0, prb1, st = run_face_detection(images_mtcnn, combo_fc, selected, selected_mtcnn, **kwargs)
        
        if return_boxes:
            return boxes0, boxes1, st
        
        images = [show_single_low(img, filepath, selected[i], ax=None, show=False, run_pred=False,
                                  boxes0=None if boxes0 is None else boxes0[i], 
                                  boxes1=None if boxes1 is None else boxes1[i], 
                                  pnts0=None if pnts0 is None else pnts0[i], 
                                  pnts1=None if pnts1 is None else pnts1[i], 
                                  prb0=None if prb0 is None else prb0[i], 
                                  prb1=None if prb1 is None else prb1[i], 
                                  **kwargs)
                  for i, img in enumerate(images_orig)]
    else:
        images = images_orig
    
    return images
    
    
def show_single(filepath, iframe=0, **kwargs):
    
    cap = cv2.VideoCapture(str(filepath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, iframe)
    _, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cap.release()
    
    return show_single_low(image, filepath, iframe, **kwargs)

    
def show_single_low(image, filepath, iframe, ax, show=True, boxes0=None, boxes1=None, pnts0=None, pnts1=None, run_pred=True,
                    fr_fc=False, fr_fl=False, fp_fc=False, fp_fl=False, cv_fc=False, ds_fc=False, dl_fc=False, bl_fc=False, ld_fc=False,
                    prb0=None, prb1=None):
    
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    
    fnt = ImageFont.truetype(FONT_PATH, 100, encoding="unic")
    d.text((10,10), str(iframe), fill=(255,255,255,128), font=fnt)
    
    rects = []
    
    if run_pred and fr_fc:
        face_positions = face_recognition.face_locations(image)
        
        margin = 0
        
        for fp in face_positions:
            offset = round(margin * (fp[2] - fp[0]))
            y0 = max(fp[0] - offset, 0)
            x1 = min(fp[1] + offset, image.shape[1])
            y1 = min(fp[2] + offset, image.shape[0])
            x0 = max(fp[3] - offset, 0)
            if True:
                d.rectangle(((x0, y0), (x1, y1)), fill=None, width=4, outline='red')
            else:
                rect = patches.Rectangle((x0,y0),x1-x0,y1-y0,linewidth=1,edgecolor='r',facecolor='none')
                rects.append(rect)

    if fr_fl:
        face_landmarks_list = face_recognition.face_landmarks(image)
        
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                d.line(face_landmarks[facial_feature], width=3)
    
    if fp_fc or fp_fl:
        if run_pred:
            mtcnn = get_detection_model(fp_fc = True)
            images_mtcnn = [image]
            images_mtcnn, shrink = mtcnn_scale_images(images_mtcnn)
            with torch.no_grad():
                boxes, probs, points = mtcnn.detect(images_mtcnn, landmarks=True)
            boxes = boxes[0]; probs=probs[0]; pnts0=points[0]

            boxes = [bb/shrink if bb is not None else None for bb in boxes]
            pnts0 = [pp/shrink if pp is not None else None for pp in pnts0]
            
            if boxes is None:
                boxes = []
                probs = []
            
            for prob, fp in zip(probs, boxes):
                #margin = 0
                #offset = round(margin * (fp[2] - fp[0]))
                offset = 14
                x0 = max(fp[0] - offset, 0)
                y1 = min(fp[1] - offset, image.shape[0])
                x1 = min(fp[2] + offset, image.shape[1])
                y0 = max(fp[3] + offset, 0)
                if True:
                    text = '%.2f'%(prob)
#                     if fi is not None:
#                         text += ' F-%s'%(fi)
                    d.rectangle(((x0, y0), (x1, y1)), fill=None, width=4, outline='white')
                    fnt = ImageFont.truetype(FONT_PATH, 50, encoding="unic")
                    d.text((x1+10,y1), text, fill=(255,255,255,128), font=fnt)
                else:
                    rect = patches.Rectangle((x0,y0),x1-x0,y1-y0,linewidth=1,edgecolor='w',facecolor='none')
                    rects.append(rect)
    
    if boxes0 is not None:
        fp = boxes0[0]
        margin = 0
        offset = round(margin * (fp[2] - fp[0]))
        x0 = max(fp[0] - offset, 0)
        y1 = min(fp[1] + offset, image.shape[0])
        x1 = min(fp[2] + offset, image.shape[1])
        y0 = max(fp[3] - offset, 0)
        d.rectangle(((x0, y0), (x1, y1)), fill=None, width=4, outline='green')
    
    if boxes1 is not None:
        fp = boxes1[0]
        offset = round(margin * (fp[2] - fp[0]))
        x0 = max(fp[0] - offset, 0)
        y1 = min(fp[1] + offset, image.shape[0])
        x1 = min(fp[2] + offset, image.shape[1])
        y0 = max(fp[3] - offset, 0)
        d.rectangle(((x0, y0), (x1, y1)), fill=None, width=4, outline='purple')
    
    for pnts in [pnts0,pnts1]:
        if pnts is None:
            continue
    #if fp_fl:
        margin = 10
        for face_landmarks in pnts:
            #d.point(face_landmarks)
            for pt in face_landmarks:
                d.ellipse([pt[0]-margin,pt[1]-margin,pt[0]+margin,pt[1]+margin])
    
    for i, (prb,box) in enumerate(zip([prb0,prb1], [boxes0, boxes1])):
        if prb is None:
            continue
        y1 = min(box[0][1] - offset, image.shape[0])
        x1 = min(box[0][2] + offset, image.shape[1])
        text = 'F%d-%.2f'%(i,prb)
        fnt = ImageFont.truetype(FONT_PATH, 50, encoding="unic")
        d.text((x1+10,y1), text, fill=(255,255,255,128), font=fnt)

    if cv_fc:
        frontal_cascade_path= str(PATH/'haarcascade_frontalface_default.xml')
        faceCascade=cv2.CascadeClassifier(frontal_cascade_path)
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.4, minNeighbors=6, minSize=(60,60))
        for x, y, w, h in faces:
            if True:
                d.rectangle(((x, y), (x+w, y+h)), fill=None, width=4, outline='yellow')
            else:
                rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='y',facecolor='none')
                rects.append(rect)
    
    if ds_fc:
        if run_pred:
            dsfd_detector = get_detection_model(ds_fc = True)
            detections = dsfd_detector.detect_face(image, confidence_threshold=.3, shrink=0.2)
            
            for x1, y1, x2, y2, p in detections:
                d.rectangle(((x1, y1), (x2, y2)), fill=None, width=4, outline='pink')
    
    if show:
        ax.axis('off')
        ax.set_title(path2name(filepath))
        ax.imshow(pil_image)
        for rect in rects:
            ax.add_patch(rect)
    
    return pil_image


# In[75]:


def show(filepaths, **kwargs):
    
    if type(filepaths) != list:
        fig, ax = plt.subplots(1, figsize=(20, 10))
        show_single(filepaths, ax=ax, **kwargs)
    else:
        rows = len(filepaths)//2
        figs, axes = plt.subplots(rows, 2, figsize=(20, 7*rows))
        for i, ax in enumerate(axes.flatten()):
            show_single(filepaths[i], ax=ax, **kwargs)


# In[76]:


def animation(filepath, n=IMAGES_PER_VIDEO, images=None, **kwargs):
    
    fig, ax = plt.subplots(1,1, figsize=(10,7))
    st = time.time()
    if images is None:
        images = get_images(filepath, n=n, **kwargs)
    print('collected images', time.time()-st)
    
    def update(frame_number):
        plt.axis('off')
        plt.imshow(images[frame_number])

    st = time.time()
    animation = FuncAnimation(fig, update, interval=100, repeat=True, frames=len(images))
    html = HTML(animation.to_jshtml())
    print('created animation', time.time()-st)
    
    return html


# In[ ]:





# In[77]:


def play(filepath):
    
    vid1 = open(str(filepath),'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(vid1).decode()
    return HTML("""<video width=600 controls><source src="%s" type="video/mp4"></video>""" % data_url)


# In[78]:


def path2name(filepaths):

    if type(filepaths) == list:
        return [f.split(splitter)[-1] for f in filepaths]
    else:
        return filepaths.split(splitter)[-1]


# In[79]:


def get_meta():
    return pd.read_csv(PATH/'metadata', low_memory=False)


# In[80]:


def get_features():
    return pd.read_csv(PATH/'features', low_memory=False)


# In[81]:


def extract_audio(filepath, output_dir=str(PATH/'wav')):
    
    output_format = 'wav'
    
    output_file = "{}{}{}.{}".format(output_dir,splitter,path2name(filepath)[:-4],output_format)
    command = f"{sudo}{PATH_FFMPEG}{splitter}ffmpeg -i {filepath} -ab 192000 -ac 2 -ar 44100 -vn " + output_file
    subprocess.call(command, shell=True)


# In[ ]:


def compress_video(filepath, output_dir=str(PATH_DISK/'compressed'), preset='slow', crf=23):
    
    output_format = 'mp4'
    fn = path2name(filepath)[:-4]
    output_path = "{}{}{}".format(output_dir,splitter,fn[:2])
    output_file = "{}{}{}_c{}.{}".format(output_path,splitter,fn,crf,output_format)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    command = f"{sudo}{PATH_FFMPEG}{splitter}ffmpeg -i {filepath} -c:v libx264 -preset {preset} -crf {crf} -c:a copy " + output_file
    subprocess.call(command, shell=True)


# In[ ]:





# # Not in use

# In[82]:


class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.length = self.container.get_meta_data()['nframes']
        self.fps = self.container.get_meta_data()['fps']
    
    def init_head(self):
        self.container.set_image_index(0)
    
    def next_frame(self):
        self.container.get_next_data()
    
    def get(self, key):
        return self.container.get_data(key)
    
    def __call__(self, key):
        return self.get(key)
    
    def __len__(self):
        return self.length


# In[ ]:





# # Shallow

# In[83]:


def string2numpy(string):
    if (type(string) == float) or (string.find('...') > 0):
        return np.array([])
    return np.array(literal_eval(string.replace(', ',',').replace('[  ','[').replace('[ ','[').replace(') (',',')
                                 .replace('(','').replace(')','').replace('rectangles','')
                                 .replace('\n   ',',').replace('\n  ',',').replace('\n ',',').replace('        ',' ')
                                 .replace('    ',' ').replace('   ',' ').replace('  ',' ').replace(' ',',')))


# In[84]:


def getCurrentBatch(dataset, fold=0, ver=None):
    if ver is None:
        ver = VERSION
    sel_batch = None
    for filename in os.listdir(PATH_MODELS):
        splits = filename.split('.')
        if len(splits) < 5: continue
        if int(splits[2][1]) != fold: continue
        if int(splits[3][1:]) != dataset: continue
        if int(splits[4][1:]) != ver: continue
        if sel_batch is None:
            sel_batch = int(splits[1][1:])
        else:
            sel_batch = max(sel_batch, int(splits[1][1:]))
    return sel_batch

def modelFileName(dataset, fold=0, batch = 1, return_last = False, return_next = False, ver=None):
    if ver is None:
        ver = VERSION
    sel_batch = batch
    if return_last or return_next:
        sel_batch = getCurrentBatch(fold=fold, dataset=dataset, ver=ver)
        if return_last and sel_batch is None:
            return None
        if return_next:
            if sel_batch is None: sel_batch = 1
            else: sel_batch += 1
    
    return 'model.b{}.f{}.d{}.v{}'.format(sel_batch, fold, dataset, ver)

def normsFileName(dataset, fold=0, ver=None):
    if ver is None:
        ver = VERSION
    
    return 'norms.f{}.d{}.v{}'.format(fold, dataset, ver)


# In[85]:


def noop(x): return x
act_fun = nn.ReLU(inplace=True)

def conv_layer(ni, nf, ks=3, act=True, padding=0):
    bn = nn.BatchNorm1d(nf)
    layers = [nn.Conv1d(ni, nf, ks, padding=padding), bn]
    if act: layers.append(act_fun)
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, ni, nh):
        super().__init__()
        layers  = [conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 5, padding=2, act=False)]
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if (ni == nh) else conv_layer(ni, nh, 1, act=False)
    
    def forward(self, x): return act_fun(self.convs(x) + self.idconv(x))

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

class ResNetModel(nn.Module):
    def __init__(self, n_cont1:int, n_cont2:int):
        super().__init__()
        
        self.n_cont1 = n_cont1
        self.n_cont2 = n_cont2
        
        self.conv2D = nn.Conv2d(1,32,(1,n_cont2))
        self.bn1 = nn.BatchNorm1d(32*6)
        
        layers = []
        layers += bn_drop_lin(32*6+n_cont1, 128, bn=False, p=0, actn=act_fun)
        layers += bn_drop_lin(128, 128, bn=True, p=0, actn=act_fun)
        layers += bn_drop_lin(128, 1, bn=True, p=0, actn=None)
        self.layers = nn.Sequential(*layers)
    
    
    def forward(self, x1, x2) -> torch.Tensor:
        
        x2 = x2.unsqueeze(1)
        x2 = self.conv2D(x2)
        x2 = x2.view(x2.shape[0],-1)
        x2 = self.bn1(x2)
        x2 = act_fun(x2)
        
        x = torch.cat([x1, x2.reshape(x2.shape[0], -1)], 1)
        
        x = self.layers(x).squeeze()
        
        return x


# In[86]:


class ResNetModel2(nn.Module):
    
    def __init__(self, n_cont=0, feat_sz=512+6+4):
        super().__init__()
        
        self.n_cont = n_cont
        self.feat_sz = feat_sz
        scale = 8
        
        self.conv2D = nn.Conv2d(1,scale*32,(1,feat_sz))
        self.bn0 = nn.BatchNorm1d(scale*32)
        
        self.res1 = ResBlock(scale*32,scale*16)
        self.res2 = ResBlock(scale*16,scale*8)
        
        self.res3 = ResBlock(scale*40,scale*16)
        self.res4 = ResBlock(scale*16,scale*8)
        
        self.res5 = ResBlock(scale*48,scale*16)
        self.res6 = ResBlock(scale*16,scale*8)
        
        self.pool1 = nn.MaxPool1d(32)
        self.pool2 = nn.AvgPool1d(32)
        
        self.linear1 = nn.Linear(scale*32,scale*8)
        self.bn1 = nn.BatchNorm1d(scale*8)
        
        self.linear = nn.Linear(scale*8,1)
    
    
    def forward(self, inp) -> torch.Tensor:
        
        batch_size = inp.shape[0]
        x = inp.reshape((2*batch_size, 32, self.feat_sz))
        
        x = self.conv2D(x.unsqueeze(1)).squeeze()
        x = self.bn0(x)
        x = act_fun(x)

        x2 = self.res1(x)
        x2 = self.res2(x2)

        x3 = torch.cat([x, x2], 1)

        x3 = self.res3(x3)
        x3 = self.res4(x3)

        x4 = torch.cat([x, x2, x3], 1)

        x4 = self.res5(x4)
        x4 = self.res6(x4)
        
        x5 = torch.cat([self.pool1(x4), self.pool2(x4)], 1).squeeze()
        x5 = x5.reshape((batch_size,-1))
        
        x6 = self.linear1(x5)
        x6 = self.bn1(x6)
        x6 = act_fun(x6)
        
        x6 = self.linear(x6)
        
        return x6.squeeze()


# In[87]:


class ResNetModel3(nn.Module):
    
    def __init__(self, feat_sz=512+6+4):
        super().__init__()
        
        self.feat_sz = feat_sz
        scale = 8
        
        self.conv2D = nn.Conv2d(1,scale*16,(1,feat_sz))

        self.bn1 = nn.BatchNorm1d(scale*16)
        self.conv1 = nn.Conv1d(scale*16,scale*16,5,stride=3) # width 10

        self.bn2 = nn.BatchNorm1d(scale*16)
        self.conv2 = nn.Conv1d(scale*16,scale*8,3) # width 8
        
        self.bn25 = nn.BatchNorm1d(scale*8)
        self.conv25 = nn.Conv1d(scale*8,scale*8,8) # width 8
        
        self.bn3 = nn.BatchNorm1d(scale*16)
        self.linear1 = nn.Linear(scale*16,scale*8)
        
        self.bn4 = nn.BatchNorm1d(scale*8)
        self.linear2 = nn.Linear(scale*8,1)
    
    
    def forward(self, inp) -> torch.Tensor:
        
        batch_size = inp.shape[0]
        x = inp.reshape((2*batch_size, 32, self.feat_sz))
        
        x = self.conv2D(x.unsqueeze(1)).squeeze()
        x = act_fun(x)

        x = self.bn1(x)
        x = self.conv1(x)
        x = act_fun(x)

        x = self.bn2(x)
        x = self.conv2(x)
        x = act_fun(x)
        
        x = self.bn25(x)
        x = self.conv25(x)
        x = act_fun(x)
        x = x.reshape((batch_size,-1))
        
        x = self.bn3(x)
        x = self.linear1(x)
        x = act_fun(x)
        
        x = self.bn4(x)
        x = self.linear2(x)
        
        return x.squeeze()


# In[88]:


class ResNetModel4(nn.Module):
    
    def __init__(self, feat_sz=512+32+8+1+2+2):
        super().__init__()
        
        self.feat_sz = feat_sz
        scale = 8
        
        self.drop = nn.Dropout(0.5)
        
        self.bn1 = nn.BatchNorm1d(feat_sz)
        self.linear1 = nn.Linear(feat_sz,128)
        
        self.bn2 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128,128)
        
        self.bn3 = nn.BatchNorm1d(256)
        #self.linear3 = nn.Linear(256,128)
        
        #self.bn4 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(256,1)
    
    
    def forward(self, inp) -> torch.Tensor:
        
        x = inp.reshape((-1, self.feat_sz))
        batch_size = int(0.5*x.shape[0])
        
        x = self.bn1(x)
        x = self.drop(x)
        x = self.linear1(x)
        x = act_fun(x)
        
        x = self.bn2(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = act_fun(x)
        
        x = x.reshape((batch_size,-1))
        x = self.bn3(x)
        x = self.drop(x)
        x = self.linear4(x)
        
        return x.squeeze()


# In[89]:


class ResNetModel5(nn.Module):
    
    def __init__(self, feat_sz=19, drop_out=0.0):
        super().__init__()
        
        self.feat_sz = feat_sz
        self.first_conv_size = 64
        self.first_sz = 32
        
        self.fea_conv = nn.Sequential(
            #nn.Dropout2d(drop_out),
            nn.Conv2d(feat_sz, self.first_conv_size, kernel_size=(1, 1), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(self.first_conv_size),
            nn.ReLU(),
            nn.Dropout2d(drop_out),
            nn.Conv2d(self.first_conv_size, self.first_sz, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(self.first_sz),
            nn.ReLU(),
            nn.Dropout2d(drop_out),
        )
        
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.first_sz, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            #nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        
        self.max_pool = nn.MaxPool1d(2*32)
        self.avg_pool = nn.AvgPool1d(2*32)
        
        self.fea_first_final = nn.Sequential(
            #nn.Linear(2*32 + 2, 1)
            nn.Linear(2, 1)
        )
        
        self.lstm_layers = 1
        self.hidden_sz = 16
        self.lstm = nn.GRU(self.first_sz, self.hidden_sz, num_layers=self.lstm_layers, batch_first=True, bidirectional=True)
        
        
        self.bottleneck_lstm = nn.Sequential(
            nn.Conv2d(2*self.hidden_sz, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            #nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        
        self.max_pool_lstm = nn.MaxPool1d(2*32)
        self.avg_pool_lstm = nn.AvgPool1d(2*32)
        
        self.fea_lstm_final = nn.Sequential(
            #nn.Linear(2*32 + 2, 1)
            nn.Linear(2, 1)
        )
    
    
    def forward(self, inp) -> torch.Tensor:
        
        # inp shape, bs x 2 x 32 x feat_sz
        inp = inp.view((-1,2,32,self.feat_sz))
        batch_size = inp.shape[0]
        x = inp.permute(0, 3, 1, 2).contiguous()
        
        x = self.fea_conv(x)
        # x shape, bs x first_sz x 2 x 32
        
        x_skip = self.bottleneck(x).squeeze() # bs x 2 x 32
        x_skip = x_skip.view(batch_size, 1, -1).contiguous()
        x_skip = torch.cat([self.max_pool(x_skip), self.avg_pool(x_skip)], 2).squeeze() #x_skip, 
        
        x_out = self.fea_first_final(x_skip)
        
        x_lstm = x.permute(0, 2, 3, 1).contiguous()
        x_lstm = x_lstm.view(2 * batch_size, 32, self.first_sz).contiguous()
        x_lstm, _ = self.lstm(x_lstm)
        # x_lstm shape, 2*bs x 32 x 2 * 32
        x_lstm = x_lstm.reshape(batch_size, 2, 32, 2*self.hidden_sz)
        x_lstm = x_lstm.permute(0, 3, 1, 2).contiguous()
        
        x_lstm = self.bottleneck_lstm(x_lstm).squeeze() # bs x 2 x 32
        x_lstm = x_lstm.view(batch_size, 1, -1).contiguous()
        
        x_lstm = torch.cat([self.max_pool_lstm(x_lstm), self.avg_pool_lstm(x_lstm)], 2).squeeze() #x_lstm, 
        x_lstm = self.fea_lstm_final(x_lstm)
        
        x_base = inp[:,0,:,12:13].mean(1)
        x_base = torch.log(x_base/(1-x_base))
        
        return (x_out + x_lstm + x_base)


# In[90]:


def get_album_transforms(mode, ds, anum):
    
    album_transforms = None

    add_trgts = dict(zip(['image' + str(i) for i in range(1,IMAGES_PER_VIDEO)], np.repeat('image',IMAGES_PER_VIDEO-1)))

    if (mode == 'test') and (ds == 1):
        #print('test augmentation num:', anum)

        
        if anum == 4:
            album_transforms = A.Compose([], additional_targets=add_trgts)
        elif anum in [0,1,2]:
            if anum == 0:
                compression = A.Compose([])
            elif anum == 1:
                compression = A.JpegCompression(quality_lower=59, quality_upper=60, p=1)
            elif anum == 2:
                compression = A.JpegCompression(quality_lower=19, quality_upper=20, p=1)
        else:
            assert False
    else:
        compression = A.OneOf([
                    A.JpegCompression(quality_lower=59, quality_upper=60, p=1),
                    A.JpegCompression(quality_lower=19, quality_upper=20, p=1),
                ], p=2/3)
    
    if album_transforms is None:
        if (mode == 'train'):
            album_transforms = A.Compose([
                    A.ShiftScaleRotate(p=0.5, scale_limit=0.15, border_mode=1, rotate_limit=15),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.GaussNoise(var_limit=20, p=0.3),
                    compression
            ], p=1, additional_targets=add_trgts)
        else:
            album_transforms = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    compression
            ], p=1, additional_targets=add_trgts)

    return album_transforms


# In[91]:


class ShallowDataSet(D.Dataset):
    
    def __init__(self, metadata, dataset, mode='train', bs=None, fold=0, ds=0, anum=-1, running_type=0):
        
        super(ShallowDataSet, self).__init__()
        
        self.mode = mode
        self.bs = bs
        self.fold = fold
        self.ds = ds
        self.anum = anum
        self.running_type = running_type
        
        self.dataset = dataset
        
        if ds==1:
            self.coupled = True
            self.take_all = False
        else:
            self.coupled = True
            self.take_all = True
        
        if mode == 'test':
            self.take_all = True
            print('ShallowDataSet test running_type', running_type, 'anum', anum)
        
        if (ds == 1) and (mode != 'test'):
            self.dataset = self.dataset.loc[self.dataset.fold == fold]
        
        if (mode != 'test') and (('original' not in dataset.columns) or ('label' not in dataset.columns)):
            self.dataset = self.dataset.join(metadata[['filename','label','original']]
                                             .set_index('filename'), on='filename')
        
        if mode in ['train','valid']:
            ds_size_original = len(self.dataset)
            self.dataset = self.dataset.loc[self.dataset.filename.isin(self.dataset.original) | 
                                            self.dataset.original.isin(self.dataset.filename)]
            print('dataset reduced for missing pairs:', mode, ds_size_original, len(self.dataset))
        
        if (mode == 'train') and (self.coupled):
            self.filenames = self.dataset.loc[self.dataset.label == 'REAL', 'filename'].unique()
        else:
            self.filenames = self.dataset.filename.unique()
        
        self.dataset = self.dataset.set_index(['filename','iframe'], drop=False)
        self.dataset.sort_index(inplace=True)
        
        if (mode == 'test') and TPU:
            batch_num = -((-len(self.filenames))//(2*MAX_DEVICES))
            samples_add = batch_num*2*MAX_DEVICES - len(self.filenames)
            self.real = np.concatenate([np.repeat(True,len(self.filenames)),np.repeat(False,samples_add)])
            
            self.filenames = np.concatenate([self.filenames, self.filenames[:samples_add]])
            print('adding samples', samples_add, 'new length', len(self.filenames))

        else:
            samples_add = 0
            self.real = np.repeat(True,len(self.filenames))
        
        self.partitions = np.random.randint(NUM_PARTITIONS, size=len(self.filenames))
        self.cur_partition = 0
        self.upd_partition()
        
        self.album_transforms = get_album_transforms(mode, ds, anum)
        
        if (mode in ['train','valid']):
            if ('weights' in self.dataset.columns) and (not self.coupled):
                self.weights = self.dataset.loc[self.dataset.iframe == 0].set_index('filename').loc[self.filenames, 'weights']
            else:
                originals = self.dataset.loc[(self.dataset.label == 'FAKE') & (self.dataset.iframe == 0), 'original']
                counts = originals.value_counts()
                if (mode == 'train') and (self.coupled):
                    self.weights = np.ones(len(self.filenames))
                    #self.weights = 1/np.sqrt(counts[originals].values)
                else:
                    self.weights = np.zeros(len(self.filenames))
                    filenames_real  = self.dataset.loc[(self.dataset.label == 'REAL') & (self.dataset.iframe == 0), 'filename']
                    assert len(filenames_real) == len(counts)
                    self.weights[[f in filenames_real for f in self.filenames]] = np.sqrt(counts[filenames_real].values)
                    self.weights[[f not in filenames_real for f in self.filenames]] = 1/np.sqrt(counts[originals].values)
                    assert np.all(self.weights != 0)
        
        self.df_time = 0
    
    def incr_partition(self):
        self.cur_partition += 1
        self.cur_partition = self.cur_partition % NUM_PARTITIONS
        self.upd_partition()

    def upd_partition(self):
        self.cur_filenames = self.filenames[self.partitions == self.cur_partition]
        if (not WEIGHTED) and (self.mode == 'train'):
            self.length = len(self.cur_filenames)
            print('new partition', self.cur_partition, 'with length', self.length)
        else:
            self.length = len(self.filenames)
            print('non-partitioned dataset with length', self.length)
    
    def get_total_len(self):
        return len(self.filenames)

    def get_feats(self, rows, filename, anum):
        
        feat = np.zeros((2, IMAGES_PER_VIDEO, 19))
        
        assert np.all(np.sort(rows.iframe.values) == rows.iframe.values)
        assert len(rows) == IMAGES_PER_VIDEO
        
        box_present = np.zeros(2, dtype=bool)
        
        for i in range(IMAGES_PER_VIDEO):
            
            mean_dim = rows.loc[i,'mean_dim']
            
            content = string2numpy(rows.loc[i,'facenet_pytorch.MTCNN.boxes0'])
            if len(content) > 0:
                feat[0,i,:4] = content/mean_dim
                box_present[0] = True

            if 'facenet_pytorch.MTCNN.boxes1' in rows.columns:
                content = string2numpy(rows.loc[i,'facenet_pytorch.MTCNN.boxes1'])
                if len(content) > 0:
                    feat[1,i,:4] = content/mean_dim
                    box_present[1] = True
            
            feat[:,i,4] = rows.loc[i,pixel_cols[0]].values

            content = string2numpy(rows.loc[i,'facenet_pytorch.MTCNN.points0'])
            if len(content) > 0:
                feat[0,i,8:18] = content.reshape(-1)/mean_dim

            
            if 'facenet_pytorch.MTCNN.points1' in rows.columns:
                content = string2numpy(rows.loc[i,'facenet_pytorch.MTCNN.points1'])
                if len(content) > 0:
                    feat[1,i,8:18] = content.reshape(-1)/mean_dim

            val = rows.loc[i,'facenet_pytorch.MTCNN.prb0']
            if type(val) == str:
                val = string2numpy(val).squeeze()
            feat[0,i,18] = 0.5 if np.isnan(val) else val

            if 'facenet_pytorch.MTCNN.prb1' in rows.columns:
                val = rows.loc[i,'facenet_pytorch.MTCNN.prb1']
                if type(val) == str:
                    val = string2numpy(val).squeeze()
                feat[1,i,18] = 0.5 if np.isnan(val) else val

        base_name = (filename[:-4]).replace(splitter,'').replace('_','').replace(':','')
        emb_file_path = PATH_WORK/'embeddings'/base_name[:2]
        #embs = np.zeros((2, IMAGES_PER_VIDEO, 512)) 
        embs = np.ones((2, IMAGES_PER_VIDEO, 512)) / np.sqrt(512)
        for i in range(2):
            emb_file = emb_file_path/(base_name + '_F' + str(i) + '_a' + str(anum))
            is_file = emb_file.is_file()
            assert (box_present[i] == is_file)
            if is_file:
                embeddings = np.loadtxt(str(emb_file))
                embs[i] = embeddings
        
        means = embs.mean(1)
        means_norm = np.linalg.norm(means, axis=1, keepdims=True)
        means /= means_norm
        
        feat[:,:,5] = (embs * np.expand_dims(means,1)).sum(2)
        
        feat[:,0,6] = 1
        feat[:,1:IMAGES_PER_VIDEO,6] = (embs[:,1:IMAGES_PER_VIDEO] * embs[:,0:IMAGES_PER_VIDEO-1]).sum(2)
        
        feat[:,:,7] = rows[['prob0_anum%d_oof'%(anum),'prob1_anum%d_oof'%(anum)]].values.transpose()
        
        assert (self.norms is not None) or (self.mode == 'train')
        
        if self.norms is not None:
            save_feat = feat[:,:,7].copy()
            feat = (feat - self.norms[0]) / self.norms[1]
            feat[:,:,7] = save_feat
        
        return feat
        
        if False:
            feat_shrunk = np.zeros((2, 512+8+1+32+2+2))

            means = feat[:,:,10:522].mean(1)
            means_norm = np.linalg.norm(means, axis=1, keepdims=True)
            means_norm = np.where(means_norm == 0, 1, means_norm)
            means /= means_norm
            feat_shrunk[:,:512] = means

            feat_shrunk[:,512] = feat[:,:,4].mean(1)

            feat_shrunk[:,513:(513+32)] = (feat[:,:,10:522] * np.expand_dims(means,1)).sum(2)
            feat_shrunk[:,513:(513+32)] = np.where(feat_shrunk[:,513:(513+32)] == 0, 1, feat_shrunk[:,513:(513+32)])

            feat_shrunk[:,(513+32):(513+32+4)] = feat[:,:,:4].mean(1)
            feat_shrunk[:,(513+32+4):(513+32+8)] = feat[:,:,:4].std(1)

            feat_shrunk[:,513+32+8] = feat_shrunk[:,513:(513+32)].mean(1)
            feat_shrunk[:,513+32+9] = feat_shrunk[:,513:(513+32)].std(1)

            feat_shrunk[:,513+32+10] = feat[:,:,522].mean(1)
            feat_shrunk[:,513+32+11] = feat[:,:,522].std(1)
            
            return feat_shrunk
        
    
    def get_faces(self, rows, filename, anum):
        
        filepath = rows.loc[0,'filepath']
        assert path2name(filepath) == filename
        faces_full, faces_crop, _ = collect_faces(filepath, df=rows)

        if self.running_type == 0:
            faces = faces_full
        else:
            faces = faces_crop
        
        for i in range(2):
            col = 'facenet_pytorch.MTCNN.boxes%d'%i
            if col in rows.columns:
                assert ((faces[i] is None) and (type(rows.loc[0,col]) == float)) or\
                       ((faces[i] is not None) and (type(rows.loc[0,col]) == str))
            else:
                assert faces[i] is None

        if not self.take_all:
            if self.mode in ['train','test']:
                sel = np.random.randint(0,32)
            else:
                sel = 10
            faces = [f[sel:(sel+1)] if f is not None else None for f in faces]
        
        if not self.take_all:
            faces = [np.stack([self.album_transforms(image=face[k].transpose((1,2,0)))['image'].transpose((2,0,1))
                            for k in range(len(face))]) if face is not None else None for face in faces]
        else:
            out = []
            for face in faces:
                if face is None:
                    out.append(None)
                    continue
                aug_input = dict(zip(['image' if i==0 else 'image' + str(i) for i in range(IMAGES_PER_VIDEO)], 
                                     list(face.transpose((0,2,3,1)))))
                res = self.album_transforms(**aug_input)
                out.append(np.stack([v.transpose((2,0,1)) for v in res.values()]))
            faces = out
        
        #np.save('C:\\StudioProjects\\DFDC\\face_temp0', faces[0])
        faces = [faces_numpy_to_tensor(face) for face in faces]
        
        return faces
    
    
    def get_faces_coupled(self, rows, filename, rows_fake, filename_fake, anum):
        
        filepath = rows.loc[0,'filepath']
        assert path2name(filepath) == filename
        filepath_fake = rows_fake.loc[0,'filepath']
        assert path2name(filepath_fake) == filename_fake

        faces, _, _ = collect_faces(filepath)
        faces_fake, _, _ = collect_faces(filepath_fake)

        if self.mode in ['train','test']:
            sel = np.random.randint(0,32)
        else:
            sel = 10
        faces = [f[sel] if f is not None else None for f in faces]
        faces_fake = [f[sel] if f is not None else None for f in faces_fake]

        assert np.all(rows.iframe.values == rows_fake.iframe.values)
        
        boxes00 = string2numpy(rows['facenet_pytorch.MTCNN.boxes0'].values[sel])
        boxes01 = string2numpy(rows_fake['facenet_pytorch.MTCNN.boxes0'].values[sel])

        boxes10 = string2numpy(rows['facenet_pytorch.MTCNN.boxes1'].values[sel])
        boxes11 = string2numpy(rows_fake['facenet_pytorch.MTCNN.boxes1'].values[sel])

        assert (faces[0] is not None) and (faces_fake[0] is not None)

        face_id = 0
        face_id_fake = 0

        if (faces[1] is not None) and (faces_fake[1] is not None):
            face_id = np.random.choice(range(2))
            face_id_fake = face_id
            if (face_id == 0):
                center = (boxes00[0] + boxes00[2])
                if np.abs(center - (boxes01[0] + boxes01[2])) > np.abs(center - (boxes11[0] + boxes11[2])):
                    face_id_fake = 1
            else:
                center = (boxes10[0] + boxes10[2])
                if np.abs(center - (boxes01[0] + boxes01[2])) < np.abs(center - (boxes11[0] + boxes11[2])):
                    face_id_fake = 0
        elif (faces[1] is not None):
            face_id_fake = 0
            face_id = 0
            center = (boxes01[0] + boxes01[2])
            if np.abs(center - (boxes00[0] + boxes00[2])) > np.abs(center - (boxes10[0] + boxes10[2])):
                face_id = 1
        elif (faces_fake[1] is not None):
            face_id = 0
            face_id_fake = 0
            center = (boxes00[0] + boxes00[2])
            if np.abs(center - (boxes01[0] + boxes01[2])) > np.abs(center - (boxes11[0] + boxes11[2])):
                face_id_fake = 1

        faces = faces[face_id]
        faces_fake = faces_fake[face_id_fake]

        if face_id == 0:
            boxes = boxes00
        else:
            boxes = boxes10
        if face_id_fake == 0:
            boxes_fake = boxes01
        else:
            boxes_fake = boxes11

        boxes_mixed = 0.5*(boxes + boxes_fake)

        if self.running_type == 0:
            im_crop = faces
            im_crop_fake = faces_fake
        else:
            im_crop = crop_in_box(boxes_mixed, faces.transpose((1,2,0)), boxes)
            im_crop_fake = crop_in_box(boxes_mixed, faces_fake.transpose((1,2,0)), boxes_fake)

        faces_both_crop = np.stack([im_crop, im_crop_fake])

        aug_input = dict(zip(['image', 'image1'], list(faces_both_crop.transpose((0,2,3,1)))))
        res = self.album_transforms(**aug_input)
        out = np.stack([v.transpose((2,0,1)) for v in res.values()])

        faces = [faces_numpy_to_tensor(face) for face in out]
        
        return faces
    

    def __getitem__(self, index):
        
        st = time.time()

        if (not WEIGHTED) and (self.mode == 'train'):
            filename = self.cur_filenames[index]
        else:
            filename = self.filenames[index]
        
        row = self.dataset.loc[filename].reset_index(drop=True)
        
        if (self.mode == 'train') and self.coupled:
            func = self.get_feats if self.ds == 0 else self.get_faces_coupled
        else:
            func = self.get_feats if self.ds == 0 else self.get_faces
        
        idx = index
        if not self.real[index]: 
            idx = -1
        if (self.mode == 'test') and (self.ds == 1):
            idx = np.repeat(idx, 2*32)
        
        if (self.ds == 0) and (self.mode in ['train','valid']):
            anum = np.random.randint(3)
        else:
            anum = self.anum

        if self.running_type == 0:
            width = 290
            height = 370
        else:
            width = 160
            height = 160
        
        if (self.mode == 'train') and (self.coupled):
            selected_fake = np.random.choice(self.dataset.loc[self.dataset.original == filename, 'filename'].values)
            row_fake = self.dataset.loc[selected_fake].reset_index(drop=True)
            self.df_time += time.time()-st
            
            assert np.all(row.label.values == 'REAL')
            assert np.all(row_fake.label.values == 'FAKE')
            
            if self.ds == 0:
                feats = [func(row, filename, anum), func(row_fake, selected_fake, anum)]
            else:
                feats = func(row, filename, row_fake, selected_fake, anum)
            
            if self.ds == 0:
                targets = np.array([0.0, 1.0])
                feats = np.stack(feats)
            else:
                # targets = [[float(k==1) if f is not None else float(-1) for f in ff] 
                #            for k,ff in enumerate(feats)]
                targets = np.array([0.0, 1.0])
                #num_imgs = 32 if self.take_all else 1
                #targets = np.repeat(np.expand_dims(np.array(targets),2), num_imgs, 2)
                # feats = [torch.stack([f if f is not None else torch.zeros((num_imgs,3,height,width)) for f in ff]) 
                #          for ff in feats]
                feats = torch.stack(feats)
                
                #flip_indices = torch.BoolTensor(size = (2,32)).random_(0, 2)
                #feats[:,:] = feats[:,:,::-1]
            
        else:
            
            feats = func(row, filename, anum)
            
            if self.mode != 'test':
                targets = float(row.loc[0,'label'] == 'FAKE')
            else:
                targets = 0.0
            
            if self.ds == 1:
                num_imgs = 32 if self.take_all else 1
                if self.mode != 'test':
                    targets = [float(row.loc[0,'label'] == 'FAKE') if f is not None else float(-1) for f in feats] 
                    targets = np.repeat(np.expand_dims(np.array(targets),1), num_imgs, 1)
                feats = torch.stack([f if f is not None else torch.zeros((num_imgs,3,height,width)) for f in feats])
                
                # if np.random.rand() > 0.5:
                #     feats = torch.flip(feats,[4])
        
        if (not CLOUD) and (self.ds == 1) and (self.take_all):
            feats = feats[:,:,:4]
            if self.mode == 'train':
                targets = targets[:,:,:4]
        
        #torch.save(feats,PATH_DISK/'temp'/filename)
        
        return feats, targets, idx
    
    
    def __len__(self):
        ret = self.length if not DATA_SMALL else int(0.01*self.length)
        #print(self.mode, 'dataset length taken', ret)
        return ret
    
    def setNorms(self, norms):
        self.norms = norms.numpy() if norms is not None else None


# In[92]:


def train_loop_fn(model, loader, ds, device, optimizer, scheduler, context=None, use_mixup=False):
    
    if CLOUD and (not CLOUD_SINGLE) and TPU:
        tlen = len(loader._loader._loader)
        OUT_TIME = 50
        generator = loader
        device_num = int(str(device)[-1])
        dataset = loader._loader._loader.dataset
    else:
        tlen = len(loader)
        OUT_TIME = 50
        generator = enumerate(loader)
        device_num = 1
        dataset = loader.dataset
    
    print('Start training {}'.format(device), 'batches', tlen)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    model.train()
    
    tloss = 0
    tloss_count = 0
    
    st = time.time()
    for i, (x, y, idx) in generator:
        
        if (not CLOUD) or CLOUD_SINGLE or (not TPU):
            if ds == 1:
                sz = x.shape
                x = x.reshape((-1,3,sz[-2],sz[-1]))
            
            if ds == 0:
                sz = x.shape
                x = x.reshape((-1,sz[-3],sz[-2],sz[-1]))

            x = x.to(device)
            y = y.reshape(-1).to(device)
        
        optimizer.zero_grad()
        
        if ds == 1:
            r = torch.randperm(len(y))
            y = y[r]
            x = x[r]
        
        mask = (y >= 0)
        y = torch.max(y,torch.zeros(len(y),dtype=torch.float64).to(device))
        
        if use_mixup:
            mix_len = mask.sum()
            lambd = np.ones(len(y))
            mask_np = mask.cpu().numpy()
            lambd[mask_np] = np.random.beta(0.4, 0.4, mix_len)
            lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
            lambd = torch.Tensor(lambd).to(device)
            shuffle = torch.arange(len(y)).to(device)
            shuffle[mask] = shuffle[mask][torch.randperm(mix_len)]
            
            if (ds == 1) or (ds == 0):
                x = lambd[:, None, None, None] * x + (1 - lambd[:, None, None, None]) * x[shuffle]
            else:
                x = lambd[:, None, None] * x + (1 - lambd[:, None, None]) * x[shuffle]
            
            y_shuffle = y[shuffle]
        
        output = model(x).squeeze()
        assert torch.isnan(output).any().cpu() == False
        
        if use_mixup:
            loss = (lambd*criterion(output, y) + (1 - lambd)*criterion(output, y_shuffle))
        else:
            if ds == 1:
                loss = 0.9*criterion(output, y) + 0.1*criterion(output, 1-y)
            else:
                loss = criterion(output, y)
        
        loss = (loss*mask).mean()
        
        loss.backward()
        
        tloss += len(y)*loss.cpu().detach().item()
        tloss_count += mask.cpu().detach().numpy().sum()
        
        if (CLOUD or CLOUD_SINGLE) and TPU:
            xm.optimizer_step(optimizer)
            if CLOUD_SINGLE:
                xm.mark_step()
        else:
            optimizer.step()
        
        scheduler.step()
        #print('learning rate:', scheduler.get_lr())
        
        st_passed = time.time() - st
        if (i+1)%OUT_TIME == 0 and device_num == 1:
            print('Batch {} device: {} time passed: {:.3f} time per batch: {:.3f}'
                .format(i+1, device, st_passed, st_passed/(i+1)))
        
        del loss, output, y, x
    
    return tloss, tloss_count

@torch.no_grad()
def val_loop_fn(model, loader, ds, device, context = None):
    
    if CLOUD and (not CLOUD_SINGLE) and TPU:
        tlen = len(loader._loader._loader)
        OUT_TIME = 50
        generator = loader
        device_num = int(str(device)[-1])
    else:
        tlen = len(loader)
        OUT_TIME = 50
        generator = enumerate(loader)
        device_num = 1
    
    #print('Start validating {}'.format(device), 'batches', tlen)
    
    st = time.time()
    model.eval()
    
    results = []
    indices = []
    targets = []
    
    for i, (x, y, idx) in generator:
        
        if (not CLOUD) or CLOUD_SINGLE or (not TPU):
            if ds == 1:
                sz = x.shape
                x = x.reshape((-1,3,sz[-2],sz[-1]))
            x = x.to(device)
        
        
        output = model(x).squeeze()
        assert torch.isnan(output).any().cpu() == False
        output = torch.sigmoid(output)
        assert torch.isnan(output).any().cpu() == False
        
        mask = (idx >= 0)
        
        if len(idx) == len(output):
            results.append(output[mask].cpu().detach().numpy())
            targets.append(y[mask].cpu().detach().numpy().reshape(-1))
        else:
            results.append(output.cpu().detach().numpy())
            targets.append(y.cpu().detach().numpy().reshape(-1))
        
        indices.append(idx[mask].cpu().detach().numpy())
        
        st_passed = time.time() - st
        if (i+1)%OUT_TIME == 0 and device_num == 1:
            print('Batch {} device: {} time passed: {:.3f} time per batch: {:.3f}'
                  .format(i+1, device, st_passed, st_passed/(i+1)))
        
        del output, y, x, idx
    
    results = np.concatenate(results)
    indices = np.concatenate(indices)
    targets = np.concatenate(targets)
    
    return results, indices, targets

@torch.no_grad()
def test_loop_fn(model, loader, device, context = None, ds = None):
    
    if CLOUD and (not CLOUD_SINGLE) and TPU:
        tlen = len(loader._loader._loader)
        OUT_TIME = 50
        generator = enumerate(loader)
        device_num = int(str(device)[-1])
        ds = 1
    else:
        tlen = len(loader)
        OUT_TIME = 50
        generator = enumerate(loader)
        device_num = 1
    
    #print('Start testing', device, 'batches', tlen, 'dataset', ds, 'context', context, 'device_num', device_num)
    
    st = time.time()
    model.eval()
    
    results = []
    indices = []
    #targets = []
    
    for i, (x, y, idx) in generator:
        
        if (not CLOUD) or CLOUD_SINGLE or (not TPU):
            x = x.to(device)
            #idx = idx.to(device)
        
        if ds == 1:
            sz = x.shape
            x = x.reshape((-1,3,sz[-2],sz[-1]))
            idx = idx.reshape((-1))
        
        output = torch.sigmoid(model(x))
        
        if len(idx) == len(output):
            mask = (idx >= 0)
            if mask.sum() > 0:
                results.append(output[mask].cpu().detach().numpy())
                indices.append(idx[mask].cpu().detach().numpy())
                #targets.append(y[mask].cpu().detach().numpy().reshape(-1))
        else:
            results.append(output.cpu().detach().numpy())
            indices.append(idx.cpu().detach().numpy())
            #targets.append(y.cpu().detach().numpy().reshape(-1))
        
        st_passed = time.time() - st
        if (i+1)%OUT_TIME == 0 and device_num == 1:
            print('B{} -> time passed: {:.3f} time per batch: {:.3f}'.format(i+1, st_passed, st_passed/(i+1)))
        
        del output, x, y, idx

    #return [],[]
    return np.concatenate(results), np.concatenate(indices)


# In[93]:


def train_one(dataset=0, epochs=1, bs=bs, fold=0, init_ver=None, freeze_to=None, meta=None, df=None, df_val=None, 
              lr=learning_rate, use_mixup=None):
    
    st0 = time.time()
    #dataset_name, filename_add, filename_add2, feat_sz,_,_,_,_ = getDSParams(dataset)
    
    cur_epoch = getCurrentBatch(fold=fold, dataset=dataset)
    if cur_epoch is None: cur_epoch = 0
    print('completed epochs:', cur_epoch, 'starting now:', epochs)
    
    #model = ResNetModel(n_cont1=144, n_cont2=513).double()
    if dataset == 0:
        model = ResNetModel5().double()
    else:
        #model = InceptionResnetDFDC(pretrained='vggface2')
        #model = models.resnet18(pretrained=False, num_classes=1)
        model = models.resnet50(pretrained=False, num_classes=1)
    
    model_file_name = modelFileName(return_last=True, fold=fold, dataset=dataset)
    
    if (model_file_name is None) and (init_ver is not None):
        model_file_name = modelFileName(return_last=True, fold=fold, dataset=dataset, ver=init_ver)
    
    if model_file_name is not None:
        print('loading model', model_file_name)
        state_dict = torch.load(PATH_MODELS/model_file_name)
        model.load_state_dict(state_dict)
    else:
        print('starting from scratch')
        if dataset == 1:
            state_dict = models.utils.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
            #state_dict = models.utils.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
            state_dict['fc.weight'] = state_dict['fc.weight'][:1]
            state_dict['fc.bias'] = state_dict['fc.bias'][:1]
            model.load_state_dict(state_dict)
    
    if freeze_to is not None:
        freeze_until(model, freeze_to)
    
    if (not CLOUD) or CLOUD_SINGLE or (not TPU):
        model = model.to(device)
    else:
        model_parallel = dp.DataParallel(model, device_ids=devices)
    
#     df_val = pd.read_csv(filepaths[-1])
#     print('validation chunk', filepaths[-1])
    
    trn_ds = ShallowDataSet(metadata=meta, dataset=df, mode='train', bs=bs, fold=fold, ds=dataset)
    val_ds = ShallowDataSet(metadata=meta, dataset=df_val, mode='valid', bs=bs, fold=fold, ds=dataset)
    
    if WEIGHTED:
        print('WeightedRandomSampler with length', int(LOADER_SCALE*len(trn_ds)), 'out of', len(trn_ds.weights))
        sampler = D.sampler.WeightedRandomSampler(trn_ds.weights, int(LOADER_SCALE*len(trn_ds)))
        loader = D.DataLoader(trn_ds, num_workers=NUM_WORKERS, batch_size=bs, 
                              shuffle=False, drop_last=True, sampler=sampler)
    else:
        loader = D.DataLoader(trn_ds, num_workers=NUM_WORKERS, batch_size=bs, 
                              shuffle=True, drop_last=True)
    
    if dataset == 0:
        norm_file_name = normsFileName(fold=fold, dataset=dataset)
        
        if model_file_name is None:
            trn_ds.setNorms(None)
            tmp_loader = D.DataLoader(trn_ds, num_workers=0, batch_size=128, shuffle=False)
            one_batch = next(iter(tmp_loader))
            one_batch = one_batch[0]
            one_batch = one_batch.view(-1,one_batch.shape[-1])
            norm_means = one_batch.mean(0)
            norm_stds = one_batch.std(0)
            norms = torch.stack([norm_means,norm_stds])
            
            torch.save(norms, PATH_MODELS/norm_file_name)
        else:
            norms = torch.load(PATH_MODELS/norm_file_name)
        
        trn_ds.setNorms(norms)
        val_ds.setNorms(norms)

    
    loader_val = D.DataLoader(val_ds, num_workers=NUM_WORKERS, batch_size=bs, shuffle=False)
    print('dataset train:', len(trn_ds), 'valid:', len(val_ds), 'loader train:', len(loader), 'valid:', len(loader_val))
    
    if use_mixup is None:
        if dataset == 1:
            use_mixup = True
        else:
            use_mixup = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5*lr, total_steps=int(trn_ds.get_total_len()/bs))
    
    for i in range(cur_epoch+1, cur_epoch+epochs+1):
        st = time.time()
        
#         filepath = filepaths[i % (len(filepaths)-1)]
#         print('training chunk', filepath)
#         df = pd.read_csv(filepath)
        
        seed_everything(1234 + i)
        
        if CLOUD and (not CLOUD_SINGLE) and TPU:
            results = model_parallel(train_loop_fn, loader, dataset, use_mixup=use_mixup, 
                                     optimizer=optimizer, scheduler=scheduler)
            tloss, tloss_count = np.stack(results).sum(0)
            state_dict = model_parallel._models[0].state_dict()
        else:
            tloss, tloss_count = train_loop_fn(model, loader, dataset, device, use_mixup=use_mixup, 
                                               optimizer=optimizer, scheduler=scheduler)
            state_dict = model.state_dict()
        
        state_dict = {k:v.to('cpu') for k,v in state_dict.items()}
        tr_ll = tloss / tloss_count
        
        trn_ds.incr_partition()

        train_time = time.time()-st
        
        print('df_time', trn_ds.df_time, 'train_time', train_time)
        
        model_file_name = modelFileName(return_next=True, fold=fold, dataset=dataset)
        if not DATA_SMALL:
            torch.save(state_dict, PATH_MODELS/model_file_name)
        
        st = time.time()
        if CLOUD and (not CLOUD_SINGLE) and TPU:
            results = model_parallel(val_loop_fn, loader_val, dataset)
            predictions = np.concatenate([results[i][0] for i in range(MAX_DEVICES)])
            indices = np.concatenate([results[i][1] for i in range(MAX_DEVICES)])
            targets = np.concatenate([results[i][2] for i in range(MAX_DEVICES)])
        else:
            predictions, indices, targets = val_loop_fn(model, loader_val, dataset, device)
        
        val_fns = val_ds.filenames[indices]
        val_w = val_ds.weights[indices]
        if dataset == 0:
            val_targets = (val_ds.dataset.xs(0, level='iframe').loc[val_fns].label == 'FAKE').astype('uint8')
        else:
            num_imgs = 32 if val_ds.take_all else 1
            val_w = np.repeat(np.expand_dims(
                np.repeat(np.expand_dims(np.array(val_w),1), 2, 1), 2), num_imgs, 2).reshape(-1)[targets >= 0]
            val_targets = targets[targets >= 0]
            predictions = predictions[targets >= 0]
        
        ll = log_loss(val_targets, predictions, eps=1e-7) #, labels=[0,1]
        llw = log_loss(val_targets, predictions, eps=1e-7, sample_weight=val_w)
        cor = np.corrcoef(val_targets, predictions)[0,1]
        auc = roc_auc_score(val_targets, predictions)
        avr = np.average(predictions)
        avrw = np.average(predictions, weights=val_w)
        avrp = np.average(predictions, weights=(val_targets==0))
        avrn = np.average(predictions, weights=(val_targets==1))
        
        print('v{}, d{}, e{}, f{}, trn ll: {:.4f}, val ll: {:.4f}, val ll w: {:.4f}, cor: {:.4f}, '              
            'auc: {:.4f}, avr w: {:.3f}, lr: {}'              
            .format(VERSION, dataset, i, fold, tr_ll, ll, llw, cor, auc, avrw, lr))
        valid_time = time.time()-st
        
        epoch_stats = pd.DataFrame([[VERSION, dataset, i, fold, tr_ll, ll, llw, cor, auc, avr, avrp, avrn, avrw,
                                     len(trn_ds), len(val_ds), bs, train_time, valid_time,
                                     lr, weight_decay, use_mixup]],
                                   columns = 
                                    ['ver','dataset','epoch','fold','train_loss','val_loss','val_loss_w','cor','auc',
                                     'avr','avr_p','avr_n','avr_w',
                                     'train_sz','val_sz','bs','train_time','valid_time','lr','wd', 'use_mixup'
                                     ])
        
        stats_filename = PATH_DISK/'stats'/'stats.f{}.v{}'.format(fold,VERSION)
        if stats_filename.is_file():
            epoch_stats = pd.concat([pd.read_csv(stats_filename), epoch_stats], sort=False)
        if not DATA_SMALL:
            epoch_stats.to_csv(stats_filename, index=False)
    
    print('total running time', time.time() - st0)
    
    return model, predictions


# In[94]:


def inference_one(dataset=0, bs=bs, fold=0, anum=4, meta=None, df_test=None, running_type=0):
    
    st = time.time()
    #dataset_name, filename_add, filename_add2, feat_sz,_,_,_,_ = getDSParams(dataset)

    cur_epoch = getCurrentBatch(fold=fold, dataset=dataset)
    if cur_epoch is None: cur_epoch = 0
    print('completed epochs:', cur_epoch)

    assert cur_epoch > 0

    #model = ResNetModel(n_cont1=144, n_cont2=513).double()
    if dataset == 0:
        model = ResNetModel5().double()
    else:
        model = models.resnet50(pretrained=False, num_classes=1)
        #model = models.resnet18(pretrained=False, num_classes=1)
    
    model_file_name = modelFileName(return_last=True, fold=fold, dataset=dataset)
    if model_file_name is not None:
        print('loading model', model_file_name)
        state_dict = torch.load(PATH_MODELS/model_file_name)
        model.load_state_dict(state_dict)
    
    if (not CLOUD) or CLOUD_SINGLE or (not TPU):
        model = model.to(device)
    else:
        model_parallel = dp.DataParallel(model, device_ids=devices)

    seed_everything(1234 + cur_epoch + anum + 100*fold)

    tst_ds = ShallowDataSet(metadata=meta, dataset=df_test, mode='test', bs=bs, fold=fold, ds=dataset, anum=anum, running_type=running_type)
    
    if dataset == 0:
        norm_file_name = normsFileName(fold=fold, dataset=dataset)
        norms = torch.load(PATH_MODELS/norm_file_name)
        tst_ds.setNorms(norms)
    
    loader_tst = D.DataLoader(tst_ds, num_workers=NUM_WORKERS, batch_size=bs, shuffle=False)

    print('dataset test:', len(tst_ds), 'loader test:', len(loader_tst), 'anum:', anum)
    
    if CLOUD and (not CLOUD_SINGLE) and TPU:
        results = model_parallel(test_loop_fn, loader_tst)
        predictions = np.concatenate([results[i][0] for i in range(MAX_DEVICES)])
        indices = np.concatenate([results[i][1] for i in range(MAX_DEVICES)])
    else:
        predictions, indices = test_loop_fn(model, loader_tst, device, ds=dataset)
    
    test_fns = tst_ds.filenames[indices]
    
    print('test processing time:', time.time() - st)
    
    return predictions, test_fns


# In[ ]:





# # Features collection

# In[95]:


def get_images_from_video(filename, v_len = None):
    
    fn = path2name(filename)
    #fn = filename.split(splitter)[-1]
    last_success = -1
    
    try:
        
        v_cap = cv2.VideoCapture(filename)
        if v_len is None:
            video_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            video_len = v_len

        assert video_len > IMAGES_PER_VIDEO

        if (MAX_FRAME is not None) and (video_len > MAX_FRAME):
            video_len = MAX_FRAME

        imgs = []
        imgs_mtcnn = []

        selected_mtcnn = np.linspace(0,video_len-1,MTCNN_IMAGES_PER_VIDEO,dtype=int)
        selected = np.linspace(0,video_len-1,IMAGES_PER_VIDEO-4,dtype=int)
        mid_num = selected[int(IMAGES_PER_VIDEO/2)]
        selected = np.sort(np.concatenate([selected, np.array([mid_num-2,mid_num-1,mid_num+1,mid_num+2])]))

        data = []
        for j in range(video_len):
            success = v_cap.grab()
            assert (fn != 'ahjnxtiamx.mp4') or (j != 200) or (not KAGGLE)

            if (j in selected) or (j in selected_mtcnn):
                success, vframe = v_cap.retrieve()
                if not success:
                    print(filename, 'failed 3', j, vframe.size)
                vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

            if j in selected_mtcnn:
                imgs_mtcnn.append(vframe)

            if j in selected:
                df = pd.DataFrame()

                df['iframe'] = [j]
                df['filename'] = [fn]
                df['pxl_mean'] = vframe.mean()

                sz = vframe.shape
                for i, p in enumerate(sample_pixels):
                    df['sp%d'%i] = vframe[int(p[0]*sz[0]), int(p[1]*sz[1])].mean()

                data.append(df)

                #vframe = Image.fromarray(vframe)
                imgs.append(vframe)

            if (j in selected) or (j in selected_mtcnn):
                last_success = j

        assert len(imgs) == IMAGES_PER_VIDEO
        assert len(imgs_mtcnn) == MTCNN_IMAGES_PER_VIDEO

        data = pd.concat(data, sort=False).reset_index(drop=True)

        v_cap.release()

        return imgs, imgs_mtcnn, filename, data, selected, selected_mtcnn

    except:
        if (v_len is None) and (last_success >= IMAGES_PER_VIDEO):
            return get_images_from_video(filename, v_len = last_success+1)
        
        return [], [], filename, None, [], []


# In[96]:


class ParseVideoDataSet(D.Dataset):
    
    def __init__(self, filenames):
        
        super(ParseVideoDataSet, self).__init__()
        self.filenames = filenames
    
    def __getitem__(self, index):
        
        filename = self.filenames[index]
        return get_images_from_video(filename)
    
    def __len__(self):
        return len(self.filenames)


# In[97]:


def feat_loop_fn(loader, device, context=None, data_mtcnn=None, anums=range(3), resnet=None):
    
    data = []
    times = defaultdict(lambda: 0)
    
    st = time.time()
    cnt = 0
    for batch in tqdm.tqdm(loader):
        
        times['loader'] += time.time() - st

        x, x_mtcnn, filename, df0, selected, selected_mtcnn = batch[0]
        fn = filename.split(splitter)[-1]
        
        if len(x) == 0:
            print('failed 1', fn)
            err_filenames[0].append(fn)
            continue
        
        if False:
            df, time_dict = collect_feats(x, x_mtcnn, filename, device, selected, selected_mtcnn, 
                                          resnet=resnet, data_mtcnn=data_mtcnn, anums=anums)
        else:
            try:
                df, time_dict = collect_feats(x, x_mtcnn, filename, device, selected, selected_mtcnn, 
                                              resnet=resnet, data_mtcnn=data_mtcnn, anums=anums)
            except:
                print('failed 2', fn)
                err_filenames[1].append(fn)
                continue
        
        del x, x_mtcnn, filename, selected, selected_mtcnn
        
        df = pd.concat([df0,df], axis=1)
        data.append(df)
        for col in list(time_dict):
            times[col] += time_dict[col]

        cnt += 1

        # if cnt % 1000 == 0:
        #     data_tmp = pd.concat(data,sort=False) if len(data) > 0 else None
        #     data_tmp.to_csv(PATH_DISK/('features_temp%d'%(cnt)), index=False)
        #     data = []

        if cnt % 10 == 0:
            gc.collect()

        st = time.time()
    
    data = pd.concat(data,sort=False) if len(data) > 0 else None
    
    return data, times


# In[98]:


def fixed_image_standardization(image_tensor):

    # mini,_ = torch.min(image_tensor,1,keepdim=True)
    # mini,_ = torch.min(mini,2,keepdim=True)
    # mini,_ = torch.min(maxi,3,keepdim=True)
    # processed_tensor = image_tensor - mini

    # maxi,_ = torch.max(processed_tensor,1,keepdim=True)
    # maxi,_ = torch.max(maxi,2,keepdim=True)
    # maxi,_ = torch.max(maxi,3,keepdim=True)
    # processed_tensor = (2*processed_tensor - maxi) / maxi

    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


# In[99]:


def faces_numpy_to_tensor(im):
    if im is None:
        return None
    tensor = torch.Tensor(im)
    tensor = fixed_image_standardization(tensor)
    return tensor


# In[100]:


def read_faces(file_loc):
    with Image.open(file_loc) as im:
        #im = imageio.imread(file_loc)
        if im.size == (1,1):
            img = None
        else:
            img = np.array(im).reshape((32,2*BOX_SIZE[1],2*BOX_SIZE[0],3)).transpose((0,3,1,2))
    return img


# In[101]:

def crop_in_box(box, face, box_center = None):
    
    if box_center is None:
        box_center = box
    
    cc = [0.5*(box_center[0] + box_center[2]), 0.5*(box_center[1] + box_center[3])]
    new_box = box - np.array([cc[0] - BOX_SIZE[0], cc[1] - BOX_SIZE[1], cc[0] - BOX_SIZE[0], cc[1] - BOX_SIZE[1]])
    new_box = new_box.astype(int)
    new_box = np.maximum(new_box, np.zeros(4,int))
    new_box = np.minimum(new_box, np.array([2*BOX_SIZE[0], 2*BOX_SIZE[1], 2*BOX_SIZE[0], 2*BOX_SIZE[1]]).astype(int))

    if ((new_box[2] - new_box[0]) < 40) or ((new_box[3] - new_box[1]) < 40):
        im = cv2.resize(face,
                        (160, 160), interpolation=cv2.INTER_AREA).copy().transpose((2,0,1))
    else:
        im = cv2.resize(face[new_box[1]:new_box[3], new_box[0]:new_box[2]],
                        (160, 160), interpolation=cv2.INTER_AREA).copy().transpose((2,0,1))
    
    return im


def collect_faces(filename, imgs=None, selected=None, selected_mtcnn=None, df=None, data=None, imgs_mtcnn=None):
    
    #fn = filename.split(splitter)[-1]

    time_dict = defaultdict()

    st = time.time()

    fn = path2name(filename)
    base_name = (fn[:-4]).replace(splitter,'').replace('_','').replace(':','')
    file_path = PATH_WORK/FACES_FOLDER/base_name[:2]
    file = [file_path/(base_name + '_F%d.jpg'%(k)) for k in range(2)]
    
    if file[0].is_file() and file[1].is_file():
        
        faces_both = [read_faces(file[0]), read_faces(file[1])]
        faces_both_crop = []
        
        if (df is not None):
            boxes0 = [string2numpy(v) for v in df['facenet_pytorch.MTCNN.boxes0'].values]
            boxes1 = [string2numpy(v) for v in df['facenet_pytorch.MTCNN.boxes1'].values]
            assert len(boxes0) == IMAGES_PER_VIDEO
            assert len(boxes1) == IMAGES_PER_VIDEO
            
            for faces, boxes in zip(faces_both, [boxes0, boxes1]):
                if faces is None:
                    faces_both_crop.append(None)
                else:
                    im_crop = np.stack([crop_in_box(boxes[k], faces[k].transpose((1,2,0))) for k in range(len(faces))])
                    faces_both_crop.append(im_crop)
        
        return faces_both, faces_both_crop, time_dict
    
    file_path.mkdir(parents=True, exist_ok=True)
    time_dict['get_images_from_video_files'] = time.time() - st

    st1 = time.time()
    if imgs is None:
        imgs, imgs_mtcnn, _, _, selected, selected_mtcnn = get_images_from_video(filename)
    time_dict['get_images_from_video'] = time.time() - st1
    
    mean_dim = np.array(imgs[0].shape[:2]).mean()
    #print(mean_dim)
    
    st2 = time.time()
    if df is not None:
        #df = df.sort_values('idx2')

        boxes0 = [mean_dim*string2numpy(v) for v in df['facenet_pytorch.MTCNN.boxes0'].values]
        boxes1 = [mean_dim*string2numpy(v) for v in df['facenet_pytorch.MTCNN.boxes1'].values]

        pnts0 = [mean_dim*string2numpy(v) for v in df['facenet_pytorch.MTCNN.points0'].values]
        pnts1 = [mean_dim*string2numpy(v) for v in df['facenet_pytorch.MTCNN.points1'].values]

        prb0 = df['facenet_pytorch.MTCNN.prb0'].values.copy()
        prb1 = df['facenet_pytorch.MTCNN.prb1'].values.copy()

        assert len(boxes0) == IMAGES_PER_VIDEO
        assert len(boxes1) == IMAGES_PER_VIDEO
        assert len(pnts0) == IMAGES_PER_VIDEO
        assert len(pnts1) == IMAGES_PER_VIDEO
        assert len(prb0) == IMAGES_PER_VIDEO
        assert len(prb1) == IMAGES_PER_VIDEO

        len0 = np.array([len(b) for b in boxes0])
        if np.all(len0 == 0):
            boxes0 = None
        else:
            assert np.all(len0 == 1)
        len1 = np.array([len(b) for b in boxes1])
        if np.all(len1 == 0):
            boxes1 = None
        else:
            assert np.all(len1 == 1)
 
    else:

        if (fn in ['ajiyrjfyzp.mp4','bebfstfcgp.mp4']) and (KAGGLE):
            boxes0, boxes1, pnts0, pnts1, prb0, prb1, st = None, None, None, None, None, None, 0
        elif (fn != 'sqixhnilfm.mp4') or (not KAGGLE):
            boxes0, boxes1, pnts0, pnts1, prb0, prb1, st = run_face_detection(imgs_mtcnn, combo_fc=True, 
                                                        selected=selected, selected_mtcnn=selected_mtcnn)
        else:
            boxes0, boxes1, pnts0, pnts1, prb0, prb1, st = run_face_detection(imgs_mtcnn, ld_fc=True, combo_fc=False, 
                                                        selected=selected, selected_mtcnn=selected_mtcnn)
    time_dict['run_face_detection'] = time.time() - st2

    faces_both = []
    faces_both_crop = []

    st3 = time.time()
    time_dict['crop_and_save_crop1'] = 0
    time_dict['crop_and_save_crop2'] = 0
    time_dict['crop_and_save_save'] = 0
    time_dict['crop_and_save_resize'] = 0

    for k, (boxes, pnts, prb) in enumerate(zip([boxes0, boxes1],[pnts0,pnts1],[prb0,prb1])):

        st = time.time()

        if boxes is None:
            faces_both.append(None)
            imageio.imwrite(file[k], [[[0]]])
            continue
        
        assert np.all(np.array([len(box) == 1 for box in boxes]))
        boxes = [b for box in boxes for b in box]
        assert len(boxes) == len(imgs)
        centers = [np.array([0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])],dtype=int) for b in boxes]
        centers = np.stack(centers)

        #imgs = [np.array(img) for img in imgs]
        sz = imgs[0].shape
        
        min_dim = min([min([int(centers[i,1]-BOX_SIZE[1]), int(centers[i,0]-BOX_SIZE[0])]) for i in range(len(boxes))])
        max_dim = max([max([int(centers[i,1]+BOX_SIZE[1]-sz[0]), int(centers[i,0]+BOX_SIZE[0]-sz[1])]) for i in range(len(boxes))])
        min_dim = min([min_dim, -max_dim])
        if min_dim < 0:
            imgs = [np.pad(img, ((-min_dim, -min_dim), (-min_dim, -min_dim), (0,0))) for img in imgs]
            centers -= min_dim
        
        time_dict['crop_and_save_crop1'] += time.time() - st
        st = time.time()

        faces = [imgs[i][(centers[i,1]-BOX_SIZE[1]):(centers[i,1]+BOX_SIZE[1]),
                         (centers[i,0]-BOX_SIZE[0]):(centers[i,0]+BOX_SIZE[0])].copy() for i in range(len(boxes))]
        assert len(faces) == len(imgs)
        sz = faces[0].shape
        assert (sz[0] == 2*BOX_SIZE[1]) and (sz[1] == 2*BOX_SIZE[0])

        assert prb is not None

        if data is not None:
            data['facenet_pytorch.MTCNN.boxes%d'%k] = list(boxes)
            data['facenet_pytorch.MTCNN.prb%d'%k] = prb
            if pnts is not None:
                data['facenet_pytorch.MTCNN.points%d'%k] = list(pnts)
        
        time_dict['crop_and_save_crop2'] += time.time() - st
        st = time.time()

        #im = np.stack([f.cpu().numpy().astype('uint8') for f in faces])
        im = np.stack(faces)
        #imageio.imwrite(file[k], im.transpose((0,2,3,1)).reshape((-1,160,3)))
        imageio.imwrite(file[k], im.reshape((-1,2*BOX_SIZE[0],3)), quality=85)

        time_dict['crop_and_save_save'] += time.time() - st
        st = time.time()

        im_crop = np.stack([crop_in_box(boxes[k], faces[k]) for k in range(len(faces))])

        faces_both.append(im)
        faces_both_crop.append(im_crop)

        time_dict['crop_and_save_resize'] += time.time() - st

    if data is not None:
        data['mean_dim'] = mean_dim
    
    time_dict['crop_and_save'] = time.time() - st3

    return faces_both, faces_both_crop, time_dict


# In[102]:


def collect_feats(imgs, imgs_mtcnn, filename, device, selected, selected_mtcnn, 
                  dlib_detector=None, resnet=None, data_mtcnn=None, anums = range(2)):
    
    time_dict = defaultdict()
    #fn = filename.split(splitter)[-1]
    fn = path2name(filename)
    st = 0

    assert (fn != 'acazlolrpz.mp4') or (not KAGGLE)
    
    def process_frame(vframe, j):
        
        st = time.time()
        if run_face_recognition:
            with torch.no_grad():
                face_positions = face_recognition.face_locations(vframe)
                face_landmarks_list = face_recognition.face_landmarks(vframe)
                enc = face_recognition.face_encodings(vframe)

            df['face_recognition.face_locations'] = [face_positions]
            df['face_recognition.face_landmarks'] = [face_landmarks_list]
            df['face_recognition.face_encodings'] = [enc]
        st = time.time() - st

        return st

    st = 0
    for j, vframe in enumerate(imgs):
        st0 = process_frame(vframe, j)
        st += st0
    time_dict['face_recognition'] = st

    st = time.time()

    data = pd.DataFrame()
    
    st2 = 0
    if run_facenet_pytorch:
        
        st0 = time.time()
        df = None
        if data_mtcnn is not None:
            df = data_mtcnn.loc[fn]
        time_dict['subset_data'] = time.time() - st0
        
        st0 = time.time()
        faces_full, faces_og, time_dict2 = collect_faces(filename, imgs=imgs, df=df, data=data, imgs_mtcnn=imgs_mtcnn, 
                                                         selected=selected, selected_mtcnn=selected_mtcnn)
        time_dict['collect_faces'] = time.time() - st0
        if time_dict2 is not None:
            time_dict.update(time_dict2)
        
        for anum in anums:
            
            album_transforms = get_album_transforms('test', ds=1, anum=anum)
            
            if len(album_transforms.transforms.transforms) > 0:
                faces_in = [np.stack([album_transforms(image=face[k].transpose((1,2,0)))['image'].transpose((2,0,1))
                                      for k in range(len(face))]) if face is not None else None for face in faces_og]
            else:
                faces_in = faces_og
            
            faces_in = [faces_numpy_to_tensor(face) for face in faces_in]
            
            for k, faces in enumerate(faces_in):

                if faces is None: continue
                assert faces[0].shape == (3,160,160)

                faces = faces.to(device)

                st2_start = time.time()
                with torch.no_grad():
                    embeddings = resnet(faces).cpu().numpy()
                st2 +=  time.time() - st2_start

                base_name = (fn[:-4]).replace(splitter,'').replace('_','').replace(':','')
                file = PATH_WORK/'embeddings'/base_name[:2]
                file.mkdir(parents=True, exist_ok=True)
                np.savetxt(str(file/(base_name + '_F' + str(k) + '_a' + str(anum))), embeddings, fmt='%.4e')
               
                del faces, embeddings
                
            del faces_in
            
        del faces_og, faces_full

    time_dict['facenet_pytorch'] = time.time() - st
    time_dict['facenet_pytorch_resnet'] = st2

    st = time.time()

    if run_cascade_classifier:
        frontal_cascade_path= str(PATH/'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(frontal_cascade_path)
        faces = [face_cascade.detectMultiScale(np.array(img), 1.4, 6) for img in imgs]

        data['cascade_classifier'] = faces

    time_dict['cascade_classifier'] = time.time() - st

    st = time.time()
    
    if run_dlib:
        
        if dlib_detector is None:
            dlib_detector = get_frontal_face_detector()
        
        boxes = [dlib_detector(np.array(img)) for img in imgs]

        data['dlib'] = boxes

    time_dict['dlib'] = time.time() - st
    
    return data, time_dict


# # Inception Resnet

# In[103]:


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetDFDC(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.num_classes is None:
            raise Exception('At least one of "pretrained" or "num_classes" must be specified')
        else:
            tmp_classes = self.num_classes

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        #self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_linear = nn.Linear(1792, 1, bias=True)
        #self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        #self.logits = nn.Linear(512, tmp_classes)

        if pretrained is not None:
            load_weights(self, pretrained)

        #if self.num_classes is not None:
        #    self.logits = nn.Linear(512, self.num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x, mode):
        
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        #x = self.last_bn(x)
        #x = F.normalize(x, p=2, dim=1)
        #if self.classify:
        #    x = self.logits(x)
        return x.squeeze()


def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.

    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.

    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        features_path = 'https://drive.google.com/uc?export=download&id=1cWLH_hPns8kSfMz9kKl9PsG5aNV2VSMn'
        logits_path = 'https://drive.google.com/uc?export=download&id=1mAie3nzZeno9UIzFXvmVZrDG3kwML46X'
    elif name == 'casia-webface':
        features_path = 'https://drive.google.com/uc?export=download&id=1LSHHee_IQj5W3vjBcRyVaALv4py1XaGy'
        logits_path = 'https://drive.google.com/uc?export=download&id=1QrhPgn1bGlDxAil2uc07ctunCQoDnCzT'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    model_dir = os.path.join(get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    state_dict = {}
    for i, path in enumerate([features_path, logits_path]):
        cached_file = os.path.join(model_dir, '{}_{}.pt'.format(name, path[-10:]))
        if not os.path.exists(cached_file):
            import requests
            from requests.adapters import HTTPAdapter
            
            print('Downloading parameters ({}/2)'.format(i+1))
            s = requests.Session()
            s.mount('https://', HTTPAdapter(max_retries=10))
            r = s.get(path, allow_redirects=True)
            with open(cached_file, 'wb') as f:
                f.write(r.content)
        state_dict.update(torch.load(cached_file))

    del state_dict['last_linear.weight']
    mdl.load_state_dict(state_dict, strict=False)


def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home


# In[104]:


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


# In[105]:


class DFDCClassificationLoss(nn.Module):
    
    def __init__(self, weight=None):
        super(DFDCClassificationLoss, self).__init__()
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss(reduction = 'none')
    
    def forward(self, x, fake, identity):
        
        probs = self.logsoftmax(x)
        probs_fake = probs[:,:NUM_FAKE_CLASSES].sum(1)
        
        loss_fake = - fake * torch.log(probs_fake) - (1-fake) * torch.log(1-probs_fake)
        loss_identity = 2*((1-fake)*self.nllloss(probs[:,NUM_FAKE_CLASSES:], identity)).mean()
        
        return loss_fake, loss_identity


# In[106]:


class ClassificationDataSet(D.Dataset):
    
    def __init__(self, metadata, dataset, mode='train', bs=None, fold=0):
        
        super(ClassificationDataSet, self).__init__()
        
        self.dataset = dataset.loc[dataset.label == 'FAKE']
        self.filenames = self.dataset.filename.unique()
        self.length = len(self.dataset)
        
        self.dataset = dataset.set_index(['filename','iframe'], drop=False)
        self.dataset.sort_index(inplace=True)
        
        self.metadata = metadata.loc[metadata.filename.isin(dataset.filename)].set_index('filename', drop=True)
        self.metadata.sort_index(inplace=True)
        
        self.mode = mode
        self.bs = bs
        self.fold = fold
        
        samples_add = 0
        self.real = np.concatenate([np.repeat(True,self.length),np.repeat(False,samples_add)])
        
    def __getitem__(self, index):
        
        st = time.time()
        filename = self.filenames[index]
        rows = self.dataset.loc[filename].reset_index(drop=True)
        target = float(self.metadata.loc[filename, 'label'] == 'FAKE')
        
        if target == 0:
            self.zeros += 1
        else:
            self.ones += 1
        
        feat = np.zeros((2, IMAGES_PER_VIDEO, 512+6+4))
        
        assert np.all(np.sort(rows.iframe.values) == rows.iframe.values)
        assert len(rows) == IMAGES_PER_VIDEO
        
        for i in range(IMAGES_PER_VIDEO):
            
            content = string2numpy(rows.loc[i,'facenet_pytorch.MTCNN.boxes0'])
            if len(content) > 0:
                feat[0,i,:4] = content
            content = string2numpy(rows.loc[i,'facenet_pytorch.MTCNN.boxes1'])
            if len(content) > 0:
                feat[1,i,:4] = content
            
            feat[:,i,4:10] = rows.loc[i,pixel_cols].values
        
        base_name = (filename[:-4]).replace(splitter,'').replace('_','').replace(':','')
        emb_file_path = PATH_WORK/'embeddings'/base_name[:2]
        for i in range(2):
            emb_file = emb_file_path/(base_name + '_F' + str(i))
            if emb_file.is_file():
                embeddings = np.loadtxt(str(emb_file))
                feat[i,:,10:] = embeddings
        
        feat_shrunk = np.zeros((2, 512+8+1+32+2))
        
        means = feat[:,:,10:].mean(1)
        means_norm = np.linalg.norm(means, axis=1, keepdims=True)
        means_norm = np.where(means_norm == 0, 1, means_norm)
        means /= means_norm
        feat_shrunk[:,:512] = means
        
        feat_shrunk[:,512] = feat[:,:,4].mean(1)
        
        feat_shrunk[:,513:(513+32)] = (feat[:,:,10:] * np.expand_dims(means,1)).sum(2)
        feat_shrunk[:,513:(513+32)] = np.where(feat_shrunk[:,513:(513+32)] == 0, 1, feat_shrunk[:,513:(513+32)])
        
        feat_shrunk[:,(513+32):(513+32+4)] = feat[:,:,:4].mean(1)
        feat_shrunk[:,(513+32+4):(513+32+8)] = feat[:,:,:4].std(1)
        
        feat_shrunk[:,513+32+8] = feat_shrunk[:,513:(513+32)].mean(1)
        feat_shrunk[:,513+32+9] = feat_shrunk[:,513:(513+32)].std(1)
        
        idx = index
        if not self.real[index]: idx = -1
        
        return feat_shrunk, target, idx
    
    
    def __len__(self):
        return self.length if not DATA_SMALL else int(0.01*self.length)


# In[ ]:

# Yuvals
import albumentations
def get_strips_df(path,min_size=1000):
    names=glob.glob(str(path/'faces/*/*.jpg'))
    sizes=[os.path.getsize(f) for f in names]
    file_df=pd.DataFrame(columns=['full_name','name','size'])
    file_df['full_name']=names
    file_df['size']=sizes
    file_df['name']=file_df['full_name'].str.split(splitter).str[-1].str[:-7]+'.mp4'
    file_df['person']=file_df['full_name'].str.split(splitter).str[-1].str[-5]
    file_df=file_df[file_df['size']>1000]
    return file_df    

class CombinedModel(nn.Module):
    
    def __init__(self, stage1,stage2,image_size=(185,145),stage1_output=False,drop_out=None,l2_norm=False):
        super().__init__()
        self.stage1=stage1
        self.stage2=stage2
        self.image_size=image_size
        self.stage1_output=stage1_output
        if self.stage1_output:
            self.stage1_class=torch.nn.Conv2d(1, 1, (1,self.stage2.in_size),stride=(1,self.stage2.in_size), padding=(0,0))
        self.drop_out=None if drop_out is None else nn.Dropout(p=drop_out)
        self.l2_norm=l2_norm
        print('returning model')
        
    def forward(self,x):
        batch_size=x.shape[0]
        num_images=x.shape[2]//self.image_size[0]
        x = x.permute(0,2,3,1).reshape(-1,self.image_size[0],self.image_size[1],3).permute(0,3,1,2)
        x = self.stage1(x)
        if len(x.shape)<4:
            x=x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        if self.l2_norm:
            xn = torch.norm(x, p=2, dim=1).unsqueeze(-1)
            x = x.div(xn.expand_as(x))
        x = x.reshape(batch_size,num_images,-1)
        if self.drop_out is not None:
            x = self.drop_out(x)
        out = self.stage2(x)
        return tuple((out,self.stage1_class(x.unsqueeze(1)))) if self.stage1_output else out

class CombinedModel2(nn.Module):
    
    def __init__(self, stage1,stage2,image_size=(185,145),stage1_output=False,drop_out=None,l2_norm=False):
        super().__init__()
        self.stage1=stage1
        self.stage2=stage2
        self.image_size=image_size
        self.stage1_output=stage1_output
        if self.stage1_output:
            self.stage1_class=torch.nn.Conv2d(1, 1, (1,self.stage2.in_size),stride=(1,self.stage2.in_size), padding=(0,0))
        self.drop_out=None if drop_out is None else nn.Dropout(p=drop_out)
        self.l2_norm=l2_norm
        print('returning model')
        
    def forward(self,x):
        batch_size=x.shape[0]
        num_images=x.shape[2]//self.image_size[0]
        x = x.permute(0,2,3,1).reshape(batch_size,-1,self.image_size[0],self.image_size[1],3).permute(1,0,4,2,3)
        xl=[]
        for i in range(x.shape[0]):
            xl.append(self.stage1(x[i]))
        x=torch.stack(xl).transpose(1,0)
        x=x.reshape((-1,)+x.shape[2:])  
        if len(x.shape)<4:
            x=x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        if self.l2_norm:
            xn = torch.norm(x, p=2, dim=1).unsqueeze(-1)
            x = x.div(xn.expand_as(x))
        x = x.reshape(batch_size,num_images,-1)
        if self.drop_out is not None:
            x = self.drop_out(x)
        out = self.stage2(x)
        return tuple((out,self.stage1_class(x.unsqueeze(1)))) if self.stage1_output else out

class FrameModel(nn.Module):
    
    def __init__(self, base_model,image_size=(185,145),mid_width=0.7):
        super().__init__()
        self.base_model=base_model
        self.image_size=image_size
        self.mid_width = mid_width
    def forward(self,x):
        in_shape=x.size()
        in_frame=(int(in_shape[-2]*self.mid_width),int(in_shape[-1]*self.mid_width))
                  
        x1 = self.base_model(x[:,:,(in_shape[-2]-in_frame[0])//2:(in_shape[-2]+in_frame[0])//2,
                 (in_shape[-1]-in_frame[1])//2:(in_shape[-1]+in_frame[1])//2])
        assert not torch.isnan(x1).any(), "x1 has nan"
        x[:,:,(in_shape[-2]-in_frame[0])//2:(in_shape[-2]+in_frame[0])//2,
                 (in_shape[-1]-in_frame[1])//2:(in_shape[-1]+in_frame[1])//2] = \
                1e-5*torch.randn(in_shape[0],in_shape[1],in_frame[0],in_frame[1])
        x=self.base_model(x)
        assert not torch.isnan(x).any(), "x has nan"
        out = torch.stack([x,x1],dim=0).max(0)[0]
        return out

class Noop(nn.Module):
    def __init__(self):
        super(Noop, self).__init__()
    def forward(self,x):
        return x
    
class ResModelPool(nn.Module):
    def __init__(self,in_size):
        super(ResModelPool, self).__init__()
        self.conv2d1=torch.nn.Conv2d(1, 64, (9,in_size),stride=(1,in_size), padding=(4,0))
        self.bn0=torch.nn.BatchNorm1d(64)
        self.conv1d1=torch.nn.Conv1d(64, 64, 7, padding=3)
        self.bn1=torch.nn.BatchNorm1d(64)
        self.relu1=torch.nn.ReLU()
        self.conv1d2=torch.nn.Conv1d(128, 64, 5, padding=2)
        self.bn2=torch.nn.BatchNorm1d(64)
        self.relu2=torch.nn.ReLU()
        self.conv1d3=torch.nn.Conv1d(192, 64, 3, padding=1)
        self.classifier = nn.Linear(32*64,1)
        
        
    def forward(self, x):
        x = self.conv2d1(x.unsqueeze(1))
        x=F.max_pool2d(x,kernel_size=(1,x.shape[-1])).squeeze(-1)        
        x0 = self.bn0(x)
        x = self.conv1d1(x0)
        x = self.bn1(x)
        x1 = self.relu1(x)
        x = torch.cat([x0,x1],1)
        x = self.conv1d2(x)
        x = self.bn2(x)
        x2 = self.relu2(x)
        x = torch.cat([x0,x1,x2],1)
        x = self.conv1d3(x)
        x = self.relu2(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out 
    
class TransformerModel(nn.Module):
    def __init__(self,in_size,
                 dim_feedforward,
                 reduce_in=None,
                 n_heads=4,
                 n_encoders=4,
                 reduce_linear=True,
                 last_pooler=None,
                 position_embd=None):
        super(TransformerModel, self).__init__()
        self.in_size=in_size
        self.reduce_in = None if reduce_in is None else nn.Linear(self.in_size, reduce_in) if reduce_linear else \
                                    nn.MaxPool1d(kernel_size=self.in_size//reduce_in)
        self.encoder_layer =nn.TransformerEncoderLayer(in_size if reduce_in is None else reduce_in, 
                                                       n_heads, 
                                                       dim_feedforward=dim_feedforward)
#        self.decoder_layer =nn.TransformerDecoderLayer(in_size, 4, dim_feedforward=in_size)
        self.encoder=nn.TransformerEncoder(self.encoder_layer, n_encoders)
#        self.decoder=nn.TransformerDecoder(self.decoder_layer, 2)
        self.classifier = nn.Linear(self.in_size if reduce_in is None else reduce_in, 1)
        self.last_pooler=last_pooler
        
        self.position_embeddings = position_embd if  position_embd is None \
                                   else self.create_embd(position_embd, in_size if reduce_in is None else reduce_in)
        
    def forward(self, x):
        if self.reduce_in is not None:
            x = self.reduce_in(x)
            x = F.relu(x)
        if self.position_embeddings is not None:
            x = x + self.position_embeddings.unsqueeze(0).expand(x.size())
        x = self.encoder_layer(x.permute(1,0,-1))
#        x = self.decoder_layer(x)
        x = x.permute(1,0,-1)
        x = x.max(1)[0] if self.last_pooler=='max' else x.mean(1) if self.last_pooler=='mean' \
                else (x.mean()+x.max(1)[0]) if self.last_pooler=='mm' else x[:,0]
        out = self.classifier(x)
        return out   
    
    def create_embd(self,in_len,embd_size,how=None):
        if how is None or how=='linear':
            param=torch.nn.Parameter(torch.linspace(0,0.1,in_len).unsqueeze(-1).expand((in_len,embd_size)))
            param.requires_grad = True
            return param
        
    
def get_model(name,pretrained=True,stage1_output=False,stage2_model='ResModel',frame_model=None,l2_norm=False,comb=0):
    drop_out=None
    if name == 'densenet169':
        stage1 = models.densenet169(pretrained=pretrained).features
        in_size = 1024
    elif name == 'densenet121':                
        stage1 = models.densenet121(pretrained=pretrained).features
        in_size = 1024
    elif name == 'resnet34':
        stage1 = models.resnet34(pretrained=pretrained)
        stage1.fc = Noop()
        in_size = 512
    elif name == 'resnet18':
        stage1 = models.resnet18(pretrained=pretrained)
        stage1.fc = Noop()
        in_size = 512
    elif name == 'resnet101':
        stage1 = models.resnet101(pretrained=pretrained)
        stage1.fc = Noop()
        in_size = 2048
    elif name == 'resnet152':
        stage1 = models.resnet152(pretrained=pretrained)
        stage1.fc = Noop()
        in_size = 2048
    elif name == 'resnext101_32x8d':
        stage1 = models.resnext101_32x8d(pretrained=pretrained)
        stage1.fc = Noop()
        in_size = 2048
    elif name == 'inception_v3':
        stage1 = models.inception_v3(pretrained=pretrained)
        stage1.fc = Noop()
        in_size = 2048
    elif name == 'InceptionResnetV1':
        stage1=InceptionResnetV1(pretrained=pretrained)
        in_size = 512
    elif name == 'inceptionv4':
        stage1 = pretrainedmodels.__dict__['inceptionv4'](pretrained='imagenet' if pretrained else None).features
        in_size = 1536
    elif name == 'xception':
        stage1 = pretrainedmodels.__dict__['xception'](pretrained='imagenet' if pretrained else None)
        stage1.last_linear = Noop()
        in_size = 2048
    else:
        raise Exception("No stage1 model named {}".format(name))
    if frame_model is not None:
        assert (frame_model>0) and (frame_model<1.0), "frame_model must be in (0,1)"
        stage1= FrameModel(stage1,mid_width=frame_model)
    if stage2_model=='ResModel':
        stage2 = ResModelPool(in_size)
    elif stage2_model=='Transformer':
        stage2 = TransformerModel(in_size,in_size)
    elif stage2_model=='Transformer256':
        stage2 = TransformerModel(in_size,256,reduce_in=256)
        drop_out=0.5
    elif stage2_model=='Transformer512':
        stage2 = TransformerModel(in_size,128,reduce_in=512,n_heads=8)
        drop_out=0.5
    elif stage2_model=='Transformer256h8':
        stage2 = TransformerModel(in_size,256,reduce_in=256,n_heads=8)
        drop_out=0.5
    elif stage2_model=='Transformer256h8':
        stage2 = TransformerModel(in_size,256,reduce_in=256,n_heads=8)
        drop_out=0.5
    elif stage2_model=='Transformer256h8p':
        stage2 = TransformerModel(in_size,256,reduce_in=256,n_heads=8,reduce_linear=False)
        drop_out=0.5
    elif stage2_model=='Transformer256h8pmax':
        stage2 = TransformerModel(in_size,256,reduce_in=256,n_heads=8,reduce_linear=False,last_pooler='max')
        drop_out=0.5
    elif stage2_model=='Transformer256h8pmean':
        stage2 = TransformerModel(in_size,256,reduce_in=256,n_heads=8,reduce_linear=False,last_pooler='mean')
        drop_out=0.5
    elif stage2_model=='Transformer256h4pmm':
        stage2 = TransformerModel(in_size,256,reduce_in=256,n_heads=4,reduce_linear=False,last_pooler='mm')
        drop_out=0.5
    elif stage2_model=='Transformer128h8':
        stage2 = TransformerModel(in_size,128,reduce_in=128,n_heads=8)
        drop_out=0.5
    elif stage2_model=='Transformer256h8e32':
        stage2 = TransformerModel(in_size,256,reduce_in=256,n_heads=8,position_embd=32)
        drop_out=0.5
    elif stage2_model=='Lstm':
        stage2 = LstmNeuralNet(embed_size=in_size,
                               hidden_size=64,
                               output_size=1,
                               MAX_SEQUENCE_LENGTH=32,
                               dropout=0.3)
    else:
        raise Exception("No stage2 model named {}".format(stage2_model))
    print('building model')
    return CombinedModel(stage1,stage2,stage1_output=stage1_output,drop_out=drop_out,l2_norm=l2_norm) if comb!=2\
        else CombinedModel2(stage1,stage2,stage1_output=stage1_output,drop_out=drop_out,l2_norm=l2_norm)

#class InferenceFaceStripDataset(D.Dataset):
#    
#    def __init__(self, filenames,face_size=160,strip_length=32):
#        
#        super(InferenceFaceStripDataset, self).__init__()
#        self.filenames = filenames
#        self.expected_shape=(face_size*strip_length,face_size,3)
#    
#    def __getitem__(self, index):
#        
#        filename = self.filenames[index]
#        res=1
#        try:
#            img = cv2.imread(filename)
#            assert img.shape == self.expected_shape
#        except AssertionError:
#            img = np.zeros(self.expected_shape)
#            res=0
#        except:
#            print("got error:", filename,img.shape)
#            img = np.zeros(self.expected_shape)
#            res=0
#            
#        return  torch.tensor(img/255.0,dtype=torch.float).permute(-1,0,1) , torch.tensor([res])
#    
#    
#    def __len__(self):
#        return len(self.filenames)    

class InferenceFaceStripDataset(D.Dataset):
    
    def __init__(self, filenames,face_size=160,strip_length=32,cropsize='half',features=None):
        
        super(InferenceFaceStripDataset, self).__init__()
        self.filenames = filenames
        self.expected_shape=(370*strip_length,290,3)
        self.output_shape=(370*strip_length//2,290//2,3)
        self.cropsize=cropsize
        if cropsize !='half':
            assert features is not None , 'features can be None only if cropsize="half"'
            self.features_filenames=features.filename.str.split('.').str[0].values
            self.bboxes=features[['facenet_pytorch.MTCNN.boxes0','facenet_pytorch.MTCNN.boxes1']].values
    
    def __getitem__(self, index):
        
        filename = self.filenames[index]
        res=1
        try:
            img = cv2.imread(filename)
            assert img.shape == self.expected_shape
            if self.cropsize=='half':
                img=img.reshape(32,370,290,3).reshape(32,185,2,145,2,3).mean(2).mean(-2).reshape(-1,145,3).astype(np.uint8)
            else:
                base_name=filename.split(splitter)[-1].split('_')[0]
                img=img.reshape(32,370,290,3)
                nimg = np.zeros((32,185,145,3),dtype=np.uint8)
                bboxes = self.bboxes[self.features_filenames==base_name][:,int(filename.split('.')[0][-1])]
                pc=0.0
                for i in range(32):
                    bbox = string2numpy(bboxes[i])
                    h=(bbox[2]-bbox[0])*self.cropsize
                    w=(bbox[3]-bbox[1])*self.cropsize
                    pc = min(1.0,max(h/370,w/290,pc))
                for i in range(32):
                    nimg[i] = cv2.resize(img[i,185-int(pc*370/2):185+int(pc*370/2), 
                                                    145-int(pc*290/2):145+int(pc*290/2)],(145,185),interpolation = cv2.INTER_CUBIC)
                                                  
                img = nimg.reshape(-1,145,3)

            
            img_t=torch.tensor(img/255.0,dtype=torch.float).permute(-1,0,1)
        except AssertionError:
            img_t = torch.zeros(self.output_shape,dtype=torch.float).permute(-1,0,1)
            res=0
#         except:
#             print("got error:", filename,img.shape)
#             img_t = torch.zeros(self.output_shape,dtype=torch.float).permute(-1,0,1)     
#             res=0
        return  img_t , torch.tensor([res])
    
    
    def __len__(self):
        return len(self.filenames)    


class InferenceFaceStripDataset2(D.Dataset):
    
    def __init__(self, filenames,face_size=160,strip_length=32,cropsize='half',features=None,augmentation=None,norm=False):
        
        super(InferenceFaceStripDataset2, self).__init__()
        self.filenames = filenames
        self.expected_shape=(370*strip_length,290,3)
        self.output_shape=(370*strip_length//2,290//2,3)
        self.cropsize=cropsize
        self.strip_length=strip_length
        self.face_size=face_size
        self.norm=norm

        self.add_trgts = dict(zip(['image' + str(i) for i in range(1,strip_length)], np.repeat('image',strip_length-1)))
        self.img_list_ind=['image' + str(i) if i>0 else 'image' for i in range(self.strip_length)]
        self.augmentation = None if augmentation is None else A.Compose(augmentation, p=1, additional_targets=self.add_trgts)

        if cropsize !='half':
            assert features is not None , 'features can be None only if cropsize="half"'
            self.features_filenames=features.filename.str.split('.').str[0].values
            self.bboxes=features[['facenet_pytorch.MTCNN.boxes0','facenet_pytorch.MTCNN.boxes1']].values
    
    def __getitem__(self, index):
        
        filename = self.filenames[index]
        res=1
        try:
            img = cv2.imread(filename)
            assert img.shape == self.expected_shape
            if self.cropsize=='half':
                img=img.reshape(32,370,290,3).reshape(32,185,2,145,2,3).mean(2).mean(-2).reshape(-1,145,3).astype(np.uint8)
            else:
                base_name=filename.split(splitter)[-1].split('_')[0]
                img=img.reshape(32,370,290,3)
                nimg = np.zeros((32,185,145,3),dtype=np.uint8)
                bboxes = self.bboxes[self.features_filenames==base_name][:,int(filename.split('.')[0][-1])]
                pc=0.0
                for i in range(32):
                    bbox = string2numpy(bboxes[i])
                    h=(bbox[2]-bbox[0])*self.cropsize
                    w=(bbox[3]-bbox[1])*self.cropsize
                    pc = min(1.0,max(h/370.0,w/290.0))
                    nimg[i] = cv2.resize(img[i,185-int(pc*370/2):185+int(pc*370/2), 
                                                    145-int(pc*290/2):145+int(pc*290/2)],(145,185),interpolation = cv2.INTER_CUBIC)
                                                  
                img = nimg.reshape(-1,145,3)
            if self.augmentation is not None:
                img_dict = dict(zip(self.img_list_ind,
                                    img.reshape(32,185,145,3)))
                aug_imags=self.augmentation(**img_dict)
                img=np.stack([aug_imags[a] for a in self.img_list_ind]).reshape(self.output_shape)

            
            img_t=torch.tensor(img/255.0,dtype=torch.float).permute(-1,0,1)
            img_t=img_t/img_t.max() if (img_t.max()>0) and self.norm else img_t

        except AssertionError:
            img_t = torch.zeros(self.output_shape,dtype=torch.float).permute(-1,0,1)
            res=0
#         except:
#             print("got error:", filename,img.shape)
#             img_t = torch.zeros(self.output_shape,dtype=torch.float).permute(-1,0,1)     
#             res=0
        return  img_t , torch.tensor([res])
    
    
    def __len__(self):
        return len(self.filenames)    


torch.nan_to_num=lambda x,nan=0.5 : torch.where(torch.isnan(x),nan*torch.ones_like(x),x)

########## Abhishek
if run_abhishek:
    import albumentations

    from PIL import Image

    BOX_SIZE = (145, 185)


    def read_faces(file_loc, box_size):
        if not os.path.exists(file_loc):
            return np.zeros((32, 2 * box_size[1], 2 * box_size[0], 3))
        with Image.open(file_loc) as im:
            if im.size == (1,1):
                img = np.zeros((32, 2 * box_size[1], 2 * box_size[0], 3))
            else:
                img = np.array(im).reshape((32, 2 * box_size[1], 2 * box_size[0], 3))
        return img

    class DeepFakeFacesTrain:
        def __init__(self, filenames, img_height, img_width, mean, std):

            self.images = filenames

            self.img_height = img_height
            self.img_width = img_width
            self.mean = mean
            self.std = std

            self.aug = albumentations.Compose([
                    albumentations.Resize(self.img_height, self.img_width, always_apply=True),
                    albumentations.Normalize(self.mean, self.std, max_pixel_value=255.0, always_apply=True)
                ])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, item):
            faces_path = self.images[item]
            images = read_faces(faces_path, box_size=BOX_SIZE)
            images = [images[i] for i in range(np.shape(images)[0])]

            for i, im in enumerate(images):
                augmented = self.aug(image=im)["image"]
                augmented = np.transpose(augmented, (2, 0, 1)).astype(np.float32)
                images[i] = augmented

            return_json = {}

            for i in range(len(images)):
                return_json[f"image_{i+1}"] = torch.tensor(images[i], dtype=torch.float)
            return return_json

    class SEResnext50_32x4d(nn.Module):
        def __init__(self, pretrained='imagenet'):
            super(SEResnext50_32x4d, self).__init__()

            self.base_model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)

            self.l0 = nn.Linear(2048, 1)

        def forward(self, x):
            batch_size, _, _, _ = x[0].shape

            x = [self.base_model.features(j) for j in x]
            x = [F.adaptive_avg_pool2d(j, 1).reshape(batch_size, -1) for j in x]
            x = torch.mean(torch.stack(x, dim=1), dim=1)
            l0 = self.l0(x)

            return l0


    def se_resnext50_32x4d(pretrained=True):
        if pretrained == True:
            return SEResnext50_32x4d(pretrained="imagenet")
        else:
            return SEResnext50_32x4d(pretrained=None)


# # Utility

# In[107]:


def balanceProbabilities(p_in, depth = 5):
    p = p_in.copy()
    if depth > 0:
        p = balanceProbabilities(p, depth=depth-1)
    p2 = 1-p
    p /= 2*p.mean()
    p = p/(p+p2)
    print(p.mean())
    return p

def priorToZeroProbabilities(p_in, prior_scaler=0.9):
    p = p_in.copy()
    x = np.log(p/(1-p))
    x *= prior_scaler
    p = 1/(1 + np.exp(-x))
    return p

def scoreInverter(score):
    return 0.5*(1-np.sqrt(1-4*np.exp(-2*score)))

def average_predictions(pp):
    return 1/(1 + np.exp( - np.log(pp / (1 - pp)).mean(0)))


# # GBM

def collect_embeddings(fn):
    
    comps = fn.split(splitter)[-1].split('_')
    df = pd.DataFrame({'filename': comps[0]+'.mp4', 'face': int(comps[1][1]), 'anum': int(comps[2][1])}, index=[0])
    embs = np.loadtxt(str(fn))
    
    means = embs.mean(0)
    means_norm = np.linalg.norm(means, keepdims=True)
    means /= means_norm

    cos_mean = (embs * np.expand_dims(means,0)).sum(1)
    cos_neig = (embs[1:IMAGES_PER_VIDEO] * embs[0:IMAGES_PER_VIDEO-1]).sum(1)
    cos_cont = (embs[16:20] * embs[17:21]).sum(1)
    
    df['cos_mean_mean'] = cos_mean.mean()
    df['cos_mean_min'] = cos_mean.min()
    df['cos_mean_max'] = cos_mean.max()
    df['cos_mean_std'] = cos_mean.std()
    
    df['cos_neig_mean'] = cos_neig.mean()
    df['cos_neig_min'] = cos_neig.min()
    df['cos_neig_max'] = cos_neig.max()
    df['cos_neig_std'] = cos_neig.std()
    
    df['cos_cont_mean'] = cos_cont.mean()
    df['cos_cont_min'] = cos_cont.min()
    df['cos_cont_max'] = cos_cont.max()
    df['cos_cont_std'] = cos_cont.std()
    
    return df

def gbm_features_boxes(data, preds):
    
    def my_fun(x):
        x=string2numpy(x)
        return np.zeros(4) if len(x)==0 else np.array([x[2]-x[0], x[3]-x[1], 0.5*(x[2]+x[0]), 0.5*(x[3]+x[1])])

    for base_col, j in zip(['facenet_pytorch.MTCNN.boxes0', 'facenet_pytorch.MTCNN.boxes1'], [0,1]):

        st = time.time()
        feats = data[base_col].apply(my_fun)
        print(time.time()-st)

        data['feats'] = feats

        out = data.groupby('filename')['feats'].apply(lambda x: np.stack(x.values).mean(0))

        for i, name in enumerate(['width','height','x','y']):
            preds[name + '.mean' + str(j)] = out.apply(lambda x: x[i]).values

        out = data.groupby('filename')['feats'].apply(lambda x: np.stack(x.values).std(0))

        for i, name in enumerate(['width','height','x','y']):
            preds[name + '.std' + str(j)] = out.apply(lambda x: x[i]).values

        del data['feats']


def gbm_features_points(data, preds):
    
    def my_fun(x):
        x=string2numpy(x)
        return np.zeros(2) if len(x)==0 else np.array([np.linalg.norm(x[0,1] - x[0,0]), np.linalg.norm(x[0,4] - x[0,3])])

    for base_col, j in zip(['facenet_pytorch.MTCNN.points0', 'facenet_pytorch.MTCNN.points1'], [0,1]):

        st = time.time()
        feats = data[base_col].apply(my_fun)
        print(time.time()-st)

        st = time.time()
        data['feats'] = feats

        out = data.groupby('filename')['feats'].apply(lambda x: np.stack(x.values).mean(0))

        for i, name in enumerate(['eyes','mouth']):
            preds[name + '.mean' + str(j)] = out.apply(lambda x: x[i]).values

        out = data.groupby('filename')['feats'].apply(lambda x: np.stack(x.values).std(0))

        for i, name in enumerate(['eyes','mouth']):
            preds[name + '.std' + str(j)] = out.apply(lambda x: x[i]).values

        del data['feats']

        print(time.time()-st)


def gbm_features_common(data, preds):
    
    for j in range(2):
        preds['conf%d_mean'%(j)] = data.groupby('filename')['facenet_pytorch.MTCNN.prb%d'%(j)].mean().values
        preds['conf%d_std'%(j)] = data.groupby('filename')['facenet_pytorch.MTCNN.prb%d'%(j)].std().values

    preds['mean_dim'] = data.groupby('filename')['mean_dim'].mean().values
    preds['pxl_mean'] = data.groupby('filename')['pxl_mean'].mean().values


def gbm_features_probs(data, preds, randomize=False, anums=range(3), suffixes=['_inf','_oof'], calc_cols=True, data_sfx=['','_full']):
    
    preds_all = []

    for anum in anums:

        print('starting anum', anum)

        preds2 = pd.DataFrame({'filename': data.filename.unique()})
        for dsf in data_sfx:
            for j in range(2):
                if calc_cols:
                    col0 = 'prob%d_anum%d_f%d%s'%(j,anum,0,dsf)
                    col1 = 'prob%d_anum%d_f%d%s'%(j,anum,1,dsf)
                    #data['prob%d_anum%d%s'%(j,anum,'_inf')] = np.where(data.fold == 0, data[col0], data[col1])
                    data['prob%d_anum%d%s%s'%(j,anum,'_oof',dsf)] = np.where(data.fold == 0, data[col1], data[col0])
                for sfx in suffixes:
                    # if randomize:
                    #     data['prob%d_anum%d%s'%(j,anum,sfx)] = np.random.rand(len(data))
                    col = 'prob%d_anum%d%s%s'%(j,anum,sfx,dsf)
                    preds2['prob%d_max%s%s'%(j,sfx,dsf)] = data.groupby('filename')[col].max().values
                    preds2['prob%d_min%s%s'%(j,sfx,dsf)] = data.groupby('filename')[col].min().values
                    preds2['prob%d_std%s%s'%(j,sfx,dsf)] = data.groupby('filename')[col].std().values
                    preds2['prob%d_mean%s%s'%(j,sfx,dsf)] = data.groupby('filename')[col].mean().values
        preds2['anum'] = anum

        preds_all.append(pd.concat([preds.drop(columns='filename'), preds2], axis=1))

    preds_all = pd.concat(preds_all).reset_index(drop=True)
    
    return preds_all


def gbm_features_emb(emb, preds_all):
    
    emb2 = emb.loc[emb.face == 0].join(emb.loc[emb.face == 1].set_index(['filename','anum']), 
                                       on = ['filename','anum'], rsuffix='1')

    del emb2['face']
    del emb2['face1']

    emb2 = emb2.fillna(0)
    
    preds_all = preds_all.join(emb2.set_index(['filename','anum']), on=['filename','anum'])
    
    return preds_all


def gbm_normalize_covs(preds):

    for col in ['width.mean0', 'height.mean0', 'x.mean0',
       'y.mean0', 'width.std0', 'height.std0', 'x.std0', 'y.std0',
       'width.mean1', 'height.mean1', 'x.mean1', 'y.mean1', 'width.std1',
       'height.std1', 'x.std1', 'y.std1', 'eyes.mean0', 'mouth.mean0',
       'eyes.std0', 'mouth.std0', 'eyes.mean1', 'mouth.mean1', 'eyes.std1',
       'mouth.std1']:
        preds[col] = 100 * preds[col] / preds['mean_dim']
    del preds['mean_dim']

    return preds

def inference_gbm(method = 'lgb', data = None, folds=range(FOLDS_VALID)):
    
    data_filt = data.drop(meta_cols + drop_cols, axis=1, errors='ignore')
    data_filt = data_filt.reindex(sorted(data_filt.columns), axis=1)

    predictions = np.zeros((len(folds), len(data_filt)))
    
    print('columns', data_filt.columns)
    
    Xt = data_filt
    
    for k, i in enumerate(folds):
        
        print('Fold', i, 'test', len(Xt))
        
        model_file_name = modelFileName(fold=i, dataset=2, ver=VERSION)

        if method == 'lgb':
            
            model = lgb.Booster(model_file = str(PATH_MODELS/model_file_name))

            predictions[k] = model.predict(Xt, num_iteration=model.best_iteration)
        
        if method == 'cat':

            bt = Xt.prob0_mean_oof_full
            bt = priorToZeroProbabilities(bt,1.2)

            test_cat = Pool(data=Xt, baseline=np.log(bt/(1-bt)))
            
            model = CatBoostClassifier(eval_metric='Logloss', task_type='GPU')
            model.load_model(str(PATH_MODELS/model_file_name))
            x = model.predict(test_cat, prediction_type='RawFormulaVal')
            pp = 1 / (1 + np.exp(-x))
            
            predictions[k] = pp

    return predictions


# # Production code

# In[108]:


if KAGGLE:
        
    if run_facenet_pytorch and (not KAGGLE_TEST):
        #!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-1.0.1-py3-none-any.whl
        sys.path.append('/kaggle/input/detection-models-code')
        
        from facenet_pytorch_local import MTCNN, InceptionResnetV1, extract_face
        #import dsfd_pytorch_local.dsfd as dsfd
        
        Path('/kaggle/working/.torch/models/checkpoints').mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = '/kaggle/working/.torch/models'
        os.system('cp /kaggle/input/facenetpytorch/*pt /kaggle/working/.torch/models/checkpoints')

        sys.path.insert(0, "/kaggle/input/detection-models-code/light_dsfd_local/DSFDv2_r18")
        from light_dsfd_local.DSFDv2_r18.model_search import Network

    
    if run_dlib:
        get_ipython().system('pip install /kaggle/input/dlibpkg/dlib-19.19.0')
        import dlib
    if (run_yuval or run_abhishek) and (not KAGGLE_TEST):
        os.system('pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/nu')
    if run_yuval:
        import pretrainedmodels


# In[109]:


if KAGGLE and not SHORT_RUN:
    
    if run_facenet_pytorch:
        resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    
    if KAGGLE_TEST:
        #run_mode = 'special_test'
        #run_mode = 'dessa'
        run_mode = 'valid'
        #run_mode = 'test_faceforensics'
    else:
        run_mode = 'valid'
        
    
    filenames = get_filepaths(run_mode)
    print(len(filenames))
    
    dataset = ParseVideoDataSet(filenames)
    loader = D.DataLoader(dataset, collate_fn=noop, num_workers=NUM_WORKERS, batch_size=1)
    
    st = time.time()
    
    data, times = feat_loop_fn(loader, device=device, anums=[4], resnet=resnet)
    
    print('running time', time.time() - st)
        
    data.to_csv(PATH_WORK/'features', index=False)
    
    del_detection_models()
    del resnet
    torch.cuda.empty_cache()

if KAGGLE and run_yuval:
    # if  not KAGGLE_TEST:
    #     PATH_MODELS = Path('/kaggle/input/xception-transformer256h8-jpgaug092')
    file_df = get_strips_df(PATH_WORK)
    df_test = pd.read_csv(PATH_WORK/'features')

    batch_size=4
    model_name='xception'
    stage2_model='Transformer256h8'
    version_ =stage2_model+'_jpgaug{}'
    predss=[]
    is_oks=[]
#     vers=[1.0,1.0,1.0,1.0,1.1,1.1,1.1,1.1]
#     splits=[0,1,2,3,0,1,2]
    vers=[0.91,0.91,0.91,0.91]
    splits=[0,1,2,3]
    augmentation=[  A.ShiftScaleRotate(p=0.5, scale_limit=0.05, border_mode=1, rotate_limit=15),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2)]   

    for k in range(4):
#        inference_ds=InferenceFaceStripDataset2(file_df.full_name.values,
#                                                cropsize=(torch.rand(1)*0.2+0.85).item(),
#                                                features=df_test,augmentation=augmentation)
        inference_ds=InferenceFaceStripDataset2(file_df.full_name.values,
                                                cropsize=0.92,
                                                features=df_test,augmentation=None)
#        version=version_.format(vers[k])
        version=version_.format(0.92)
        split=k #splits[k%4]
        print('process split',split,' version ',version)
        model_file='{}_s{}_v{}.pth'.format(model_name,split,version)
        model = get_model(model_name,stage1_output=False,stage2_model=stage2_model,pretrained=False,l2_norm=True).eval().to(device)
#        model = pretrainedmodels.__dict__[model_name](pretrained=None,num_classes=1).to(device).eval()
        model.load_state_dict(torch.load(PATH_MODELS/model_file))
        pred_list=[]
        is_ok_list=[]
        with torch.no_grad():
            dl=D.DataLoader(inference_ds, batch_size=batch_size, shuffle=False,num_workers=NUM_WORKERS)
            tk0 = tqdm.tqdm(dl)
            for i,batch in enumerate(tk0):
                b = batch[0]
#                b = batch[0].reshape(-1,3,32,160,160).permute(0,2,1,3,4).reshape(-1,3,160,160)
                pred_list.append(model(b.to(device)).to('cpu').detach().numpy())
                is_ok_list.append(batch[1].numpy())
        predss.append(np.concatenate(pred_list))
#        predss.append(np.median(np.concatenate(pred_list).reshape(-1,32),1))
        is_oks.append(np.concatenate(is_ok_list))
    preds=np.nan_to_num(np.median(np.stack(predss),0),nan=0.5)
    is_ok=np.stack(is_oks).min(0)
    file_df['preds']=preds
    file_df['is_ok']=is_ok

    sub0 = pd.read_csv(PATH/'sample_submission.csv')

    m = sub0.merge(file_df[file_df.person=='0'][['name','preds','is_ok']],how='left',left_on='filename',right_on='name').rename(columns={'preds':'person0_pred','is_ok':'person0_is_ok'}).drop('name',axis=1)
    m = m.merge(file_df[file_df.person=='1'][['name','preds','is_ok']],how='left',left_on='filename',right_on='name').rename(columns={'preds':'person1_pred','is_ok':'person1_is_ok'}).drop('name',axis=1)
    m.person0_pred=m.person0_pred.fillna(0)
    m.person0_is_ok=m.person0_is_ok.fillna(0)
    m.person1_is_ok=m.person1_is_ok.fillna(0)
    m['pred']=np.where(m.person1_is_ok>0,m[['person0_pred','person1_pred']].max(1),m.person0_pred)
    m['label']=torch.sigmoid(torch.tensor(m.pred.values)).clamp(0.01,0.99)

    m.label.hist(bins=50)

    sub = m[['filename','label']]
    print(sub.shape)
    print(sub.head(10))

    if not run_zahar:
        sub.to_csv('submission.csv', index=False)
    else:
        sub_yuval = sub.copy()

    if len(sub) == 400:
        # if not KAGGLE_TEST:
        #     PATH_META= Path('/kaggle/input/test-meta')
        meta = pd.read_csv(PATH_META/'metadata', low_memory=False)

        sub = sub.rename(columns={'label':'pred'}).join(meta[['filename','label']].set_index('filename'), on='filename')

        from sklearn.metrics import confusion_matrix

        print(confusion_matrix((sub.label == 'FAKE').values, (sub.pred > 0.5)))
        print('mean prediction', sub.pred.mean())
        print('accuracy', ((sub.label == 'FAKE').values == (sub.pred > 0.5)).mean())
        print('log-loss', log_loss((sub.label == 'FAKE').values, sub.pred, eps=1e-7, labels=[0,1]))
        print('min: ',sub.pred.min(),' max: ',sub.pred.max())
    
    if (not KAGGLE_TEST) and (not run_zahar):
        shutil.rmtree(PATH_WORK/'embeddings')
        shutil.rmtree(PATH_WORK/'faces')


# In[110]:


if KAGGLE and run_zahar:
    
    #VERSION = 59
    
    meta = None
    
    df_test = pd.read_csv(PATH_WORK/'features')
    df_test['filepath'] = enrich_filepath(df_test.filename, run_mode)
    df_test['idx'] = df_test.groupby('filename')['iframe'].transform(lambda x: np.arange(len(x)))
    
    print(df_test['facenet_pytorch.MTCNN.boxes0'].isnull().value_counts())
    if 'facenet_pytorch.MTCNN.points0' in df_test.columns:
        print(df_test['facenet_pytorch.MTCNN.points0'].isnull().value_counts())
    if 'facenet_pytorch.MTCNN.boxes1' in df_test.columns:
        print(df_test['facenet_pytorch.MTCNN.boxes1'].isnull().value_counts())
    if 'facenet_pytorch.MTCNN.points1' in df_test.columns:
        print(df_test['facenet_pytorch.MTCNN.points1'].isnull().value_counts())

    probs_list = []

    for ver, dsfx, running_type in zip([59,58], ['_full',''], [0, 1]):

        VERSION = ver
        print('inference for ver',VERSION,'running_type',running_type)

        for fold in range(2):
            predictions, test_fns = inference_one(dataset=1, fold=fold, bs=ds1_bs, df_test=df_test, running_type=running_type)
            
            probs_df = pd.DataFrame(predictions.reshape((-1,2,32)).transpose((0,2,1)).reshape((-1,2)), 
                                    columns = ['prob0.f%d%s'%(fold,dsfx),'prob1.f%d%s'%(fold,dsfx)])
            fn_df = pd.DataFrame({'filename': test_fns.reshape((-1,2))[:,0], 'idx': np.tile(range(32),int(len(test_fns)/64))})
            probs_df = pd.concat([fn_df, probs_df], axis=1)
            
            df_test = df_test.join(probs_df.set_index(['filename','idx']), on=['filename','idx'])

    print(df_test.describe())


# In[111]:


if KAGGLE and run_zahar and False:
    
    VERSION = 59
    
    preds = []
    fns = []
    
    for fold in range(2):
    
        df_test['prob0_anum4_oof'] = df_test['prob0.f%d'%(fold)]
        df_test['prob1_anum4_oof'] = df_test['prob1.f%d'%(fold)]
        
        predictions, test_fns = inference_one(bs=ds0_bs, df_test=df_test)
        preds.append(predictions)
        fns.append(test_fns)
    
    assert np.all(fns[0] == fns[1])
    preds = np.stack(preds)
    #preds = np.exp(np.log(preds + 1e-6).mean(0))
    predictions = np.sqrt(preds[0] * preds[1]).squeeze()
    
    bb = plt.hist(predictions, bins=50)

    print('min prediction', predictions.min())
    print('max prediction', predictions.max())
    print('quantiles prediction', np.array([np.quantile(predictions,q=i) for i in np.arange(0.1,1,0.1)]))


if KAGGLE and run_zahar and True:

    VERSION = 59

    filepaths = glob.glob(str(PATH_WORK/'embeddings/*/*'))
    print('embeddings files', len(filepaths))

    st = time.time()
    result = Parallel(n_jobs=2)(delayed(collect_embeddings)(fn) for fn in tqdm.tqdm(filepaths))
    print('running time', time.time() - st)

    emb = pd.concat(result).reset_index(drop=True)

    data = df_test
    data = data.sort_values(['filename','iframe'])

    def my_fun(x):
        if x is np.nan:
            return 0
        return string2numpy(x)[0,0]

    data['facenet_pytorch.MTCNN.prb0'] = data['facenet_pytorch.MTCNN.prb0'].apply(my_fun)
    data['facenet_pytorch.MTCNN.prb1'] = data['facenet_pytorch.MTCNN.prb1'].apply(my_fun)

    assert np.all(data.groupby('filename')['filename'].apply(lambda x: x.values[0]).values ==  data.filename.unique())

    preds = pd.DataFrame({'filename': data.filename.unique()})

    gbm_features_boxes(data, preds)
    gbm_features_points(data, preds)
    gbm_features_common(data, preds)

    preds = gbm_normalize_covs(preds)

    predictions = []

    for fold in range(FOLDS_VALID):

        data['prob0_anum4_oof'] = data['prob0.f%d'%(fold)].values
        data['prob1_anum4_oof'] = data['prob1.f%d'%(fold)].values
        data['prob0_anum4_oof_full'] = data['prob0.f%d_full'%(fold)].values
        data['prob1_anum4_oof_full'] = data['prob1.f%d_full'%(fold)].values
        
        preds_all = gbm_features_probs(data, preds.copy(), anums=[4], suffixes=['_oof'], calc_cols=False)

        preds_all = gbm_features_emb(emb, preds_all)
        assert preds_all.shape[0] == data.filename.nunique()
        preds_all = preds_all.fillna(0)

        predictions.append(inference_gbm(method='cat', data=preds_all, folds=[fold]))
        #predictions.append(np.expand_dims(priorToZeroProbabilities(preds_all.prob0_mean.values, 1.2),0))

    predictions = np.concatenate(predictions,axis=0)
    predictions = average_predictions(predictions)

    # print('min prediction', predictions.min())
    # print('max prediction', predictions.max())

    #predictions = priorToZeroProbabilities(predictions, 0.85)

    test_fns = data.filename.unique()

    bb = plt.hist(predictions, bins=50)

    print('min prediction', predictions.min())
    print('max prediction', predictions.max())
    print('quantiles prediction', np.array([np.quantile(predictions,q=i) for i in np.arange(0.1,1,0.1)]))



# In[112]:


if KAGGLE and run_zahar:

    filenames_noface = df_test.loc[df_test['facenet_pytorch.MTCNN.boxes0'].isnull() & df_test['facenet_pytorch.MTCNN.boxes1'].isnull(), 'filename'].unique()
    print('filenames no face', filenames_noface)

    
    sub = pd.DataFrame({'filename': test_fns, 'pred': predictions})
    for i in range(len(err_filenames)):
        if len(err_filenames[i]) == 0:
            continue
        pred_err = 0.5
        sub_err = pd.DataFrame({'filename': err_filenames[i], 'pred': pred_err})
        sub = pd.concat([sub, sub_err], sort=False)
    
    if len(filenames_noface) > 0:
        sub.loc[sub.filename.isin(filenames_noface), 'pred'] = 0.5

    if not KAGGLE_TEST:
        sub0 = pd.read_csv(PATH/'sample_submission.csv')
        if len(sub) != len(sub0):
            sub0.label = 0.5
            sub0.to_csv('submission.csv', index=False)
            assert False
        
#         sub0 = sub0.join(sub.set_index('filename'), on='filename')
#         sub0 = sub0.fillna(0.5)
#         sub = sub0.drop(columns='label')
    
    sub['pred'] = np.clip(sub['pred'].values, 0.1, 0.99)
    sub['pred'] = balanceProbabilities(sub['pred'].values)
    
    if len(sub) == 400:
        meta = pd.read_csv(PATH_META/'metadata', low_memory=False)
        sub = sub.join(meta[['filename','label']].set_index('filename'), on='filename')

        from sklearn.metrics import confusion_matrix

        print(confusion_matrix((sub.label == 'FAKE').values, (sub.pred > 0.5)))
        print('mean prediction', sub.pred.mean())
        print('accuracy', ((sub.label == 'FAKE').values == (sub.pred > 0.5)).mean())
        print('log-loss', log_loss((sub.label == 'FAKE').values, sub.pred, eps=1e-7, labels=[0,1]))

        cuts = pd.qcut(sub.pred.values, q=np.arange(0,1.1,0.1), duplicates='drop')
        quants = pd.DataFrame(zip([sub.pred.values[cuts == c].mean() for c in cuts.unique()], 
                                  [(sub.label == 'FAKE').values[cuts == c].mean() for c in cuts.unique()]))
        print(quants.sort_values(0))

    # err_count = len(err_filenames[0]) + len(err_filenames[1])
    
    # if err_count > 0:
    #     sub['pred'] = scoreInverter(0.7 + 0.00001 * err_count)
    
    # sub['pred'] = scoreInverter(0.7 + 0.00001 * len(filenames_noface))

    sub['label'] = sub.pred
    del sub['pred']

    sub = sub.sort_values('filename')

    if run_yuval:
        sub_yuval = sub_yuval.sort_values('filename')
        sub.label = balanceProbabilities(0.2*sub.label.values + 0.8*sub_yuval.label.values)
    
    sub.to_csv('submission.csv', index=False)
    
    if not KAGGLE_TEST:
        shutil.rmtree(PATH_WORK/'embeddings')
        shutil.rmtree(PATH_WORK/'faces')

# In[113]:

if KAGGLE and run_abhishek:
    if  not KAGGLE_TEST:
        PATH_MODELS = Path('/kaggle/input/dfdc-abhishek-models')
    file_df = get_strips_df(PATH_WORK)
    df_test = pd.read_csv(PATH_WORK/'features')

    batch_size=8
    IMG_HEIGHT=64
    IMG_WIDTH=64
    MODEL_MEAN=(0.485, 0.456, 0.406)
    MODEL_STD=(0.229, 0.224, 0.225)
    inference_ds = DeepFakeFacesTrain(
        file_df.full_name.values,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    model = SEResnext50_32x4d(pretrained=None)
    model = nn.DataParallel(model,device_ids=[0])
    model.load_state_dict(torch.load(PATH_MODELS/"se_resnext50_32x4d_fold_4.bin"))
    model=model.to(device).eval()
    pred_list=[]
    with torch.no_grad():
        dl=D.DataLoader(inference_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        tk0 = tqdm.tqdm(dl)
        for bi,d in enumerate(tk0):
            images = [d[f"image_{i + 1}"] for i in range(32)]
            images = [i.to(device, dtype=torch.float) for i in images]
            outputs = model(images)
            pred_list.append(outputs.to('cpu').detach().numpy())
            
    preds=np.nan_to_num(np.concatenate(pred_list),nan=0.5)
    file_df['preds']=preds
    file_df['is_ok']=1

    sub0 = pd.read_csv(PATH/'sample_submission.csv')

    m = sub0.merge(file_df[file_df.person=='0'][['name','preds','is_ok']],how='left',left_on='filename',right_on='name').rename(columns={'preds':'person0_pred','is_ok':'person0_is_ok'}).drop('name',axis=1)
    m = m.merge(file_df[file_df.person=='1'][['name','preds','is_ok']],how='left',left_on='filename',right_on='name').rename(columns={'preds':'person1_pred','is_ok':'person1_is_ok'}).drop('name',axis=1)
    m.person0_pred=m.person0_pred.fillna(0)
    m.person0_is_ok=m.person0_is_ok.fillna(0)
    m.person1_is_ok=m.person1_is_ok.fillna(0)
    m['pred']=np.where(m.person1_is_ok>0,m[['person0_pred','person1_pred']].min(1),m.person0_pred)
    m['label']=1-torch.sigmoid(torch.tensor(m.pred.values)).clamp(0.01,0.99)

    m.label.hist(bins=50)

    sub = m[['filename','label']]
    print(sub.shape)
    print(sub.head(10))
    sub.to_csv('submission.csv', index=False)

    if len(sub) == 400:
        if not KAGGLE_TEST:
            PATH_META= Path('/kaggle/input/test-meta')
        meta = pd.read_csv(PATH_META/'test_meta.csv', low_memory=False)


        sub = sub.rename(columns={'label':'pred'}).join(meta[['filename','label']].set_index('filename'), on='filename')

        from sklearn.metrics import confusion_matrix

        print(confusion_matrix((sub.label == 'FAKE').values, (sub.pred > 0.5)))
        print('mean prediction', sub.pred.mean())
        print('accuracy', ((sub.label == 'FAKE').values == (sub.pred > 0.5)).mean())
        print('log-loss', log_loss((sub.label == 'FAKE').values, sub.pred, eps=1e-7, labels=[0,1]))
        print('min: ',sub.pred.min(),' max: ',sub.pred.max())
    if  not KAGGLE_TEST:
        shutil.rmtree(PATH_WORK/'embeddings')
        shutil.rmtree(PATH_WORK/'faces')
# In[113]:

if KAGGLE:
    print(err_filenames)


# In[ ]:




