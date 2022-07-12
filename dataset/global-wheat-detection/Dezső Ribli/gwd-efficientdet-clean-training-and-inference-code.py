"""
EfficientDet Global Wheat Detection code

Sources (big thanks to)
- https://www.kaggle.com/shonenkov/training-efficientdet
- https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
- https://www.kaggle.com/pestipeti/competition-metric-details-script
- https://github.com/rwightman/efficientdet-pytorch

Author: Dezso Ribli
"""

CONFIG_NAMES  = ['tf_efficientdet_d6']

W = ['../input/effdet6-gwd-trained/best-checkpoint-037epoch_newanchors.bin']  


################################################################################
# Kaggle/COLAB setup

import os
import sys
import subprocess

SEED = 42
PSEUDO_LABELS = False  # never happened in the end... :(s
if 'COLAB_GPU' in os.environ:
    DIR_INPUT = 'data/original' 
    # Modified effdet with YOLOv5 anchors instead of the highest scale anchors
    # It gave some small boost
    sys.path.insert(0, "efficientdet-pytorch-mod")
    sys.path.insert(0, "Weighted-Boxes-Fusion")
else:
    DIR_INPUT = '../input/global-wheat-detection/'
    # install 2 packages
    subprocess.check_output("pip install --no-deps '../input/timm-package/timm-0.1.30-py3-none-any.whl' > /dev/null", shell=True)
    subprocess.check_output("pip install --no-deps '../input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl' > /dev/null", shell=True)
    if PSEUDO_LABELS:  # install apex for pseudo labelling only (never happened)
        subprocess.check_output('cd ../input/myapex/apex && pip install --no-deps -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./', shell=True)
    # add other packages to path
    sys.path.insert(0, "../input/myeffdet/efficientdet-pytorch-mod")
    sys.path.insert(0, "../input/weightedboxesfusion")
    sys.path.insert(0, "../input/omegaconf")

    
# prevent tqdm from killing the notebook when submitting
VERBOSE = len(os.listdir(f'{DIR_INPUT}/test')) < 11 
# TTA only in real submission with many test images
TTA = len(os.listdir(f'{DIR_INPUT}/test')) > 11
    


################################################################################
# misc.py
################################################################################

import torch
import random
import numpy as np
import os
import subprocess

def seed_everything(seed):
    """Should be names seed something."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # turns out its not working together, its not deterministic in the end
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

################################################################################
# data_utils.py
################################################################################

import numpy as np
import pandas as pd
import random
from glob import glob
from sklearn.model_selection import KFold, StratifiedKFold

import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler



def load_data(dir_input):
    """Load train.csv and replace bbox string with xywh."""
    df = pd.read_csv(f'{dir_input}/train.csv')
    xywh = [map(float,x[1:-1].split(','))  for x in df.bbox]
    df[['x', 'y', 'w', 'h']]  = pd.DataFrame(xywh)
    df.drop(columns='bbox',inplace=True)

    # remove bad boxes
    # https://www.kaggle.com/c/global-wheat-detection/discussion/159578
    bad_boxes = [3687, 117344,173,113947,52868,2159,2169,121633,
                 121634,147504,118211,52727,147552]
    df.drop(bad_boxes, inplace=True)

    # it may be weird to add df 2x, but the functions is a bit more felxible
    # e.g.: when I'm only adding empty images to training df
    df = add_empty_images_to_train(df, df, dir_input)

    return df


def kfold_by_image(df, dir_input, n_splits=5, shuffle=True, random_state=42):
    """Split training data to K sets by images."""
    image_ids = df['image_id'].unique()
    
    mykfold = KFold(n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, val_idx in mykfold.split(image_ids):

        val_df = df[df['image_id'].isin(image_ids[val_idx])]
        train_df = df[df['image_id'].isin(image_ids[train_idx])]
        # train_df = add_empty_images_to_train(df, train_df, dir_input)

        yield train_df, val_df


def add_empty_images_to_train(df, train_df, dir_input):
    ### Add nan lines for images with no bounding boxes ####
    all_image_fns = os.listdir(f'{dir_input}/train/')
    all_image_ids = [os.path.splitext(fn)[0] for fn in all_image_fns]
    # find the images with no match in the DF
    empty_image_ids = sorted(set(all_image_ids).difference(set(df.image_id)))
    # make empty records for the empty images
    records_to_add = [{'image_id':im_id} for im_id in empty_image_ids]
    # concat DFs, it results in NaNs for every column in the empty images
    train_df = pd.concat([
        train_df, pd.DataFrame(records_to_add)]).reset_index(drop=True)
    return train_df



def make_test_df(dir_test, boxes = None, image_ids=None):
    records = []
    if boxes:  # make pseudolabel records
        for i in range(len(image_ids)):
            for box in boxes[i]:
                xywh = int_yxyx2xywh(box)
                records.append({
                    'image_id': image_ids[i], 
                    'path' : f'{dir_test}/{image_ids[i]}.jpg',
                    'x': xywh[0], 'y': xywh[1], 
                    'w': xywh[2], 'h': xywh[3]})

    else:  # dummy records for prediction
        fns = glob(dir_test+'/*.jpg')
        for fn in fns:
            records.append({
                'image_id': os.path.splitext(os.path.basename(fn))[0],
                 'path' : fn,
                 'x':42, 'y':42, 'w':42, 'h':42})

    return pd.DataFrame(records)



class WheatDataset(Dataset):

    def __init__(self, df, image_dir, transforms=None, min_im_size = 1024,
                 max_im_size=1024, test=False, yxyx=True, transpose_train=True, 
                 multi_scale_training=False, pad_multi_scale=False,
                 use_streched_mosaic = False, size_limit = 1):
        super().__init__()

        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.min_im_size = min_im_size
        self.max_im_size = max_im_size 
        self.test = test
        self.yxyx = yxyx
        self.transpose_train = transpose_train
        self.multi_scale_training = multi_scale_training
        self.pad_multi_scale = pad_multi_scale
        if use_streched_mosaic:
            self.mosaic_f = self.load_stretched_mosaic_image_and_boxes
        else:
            self.mosaic_f = self.load_mosaic_image_and_boxes
        # remove boxes with any edge shorter than this
        self.size_limit = size_limit 

    def __getitem__(self, index: int):      
        ### load image or mosaic images
        if self.test: # Dezso, stronger cutmix (or random.random() > 0.5:)
            image, boxes = self.load_image_and_boxes(index)
        else:
            # try to load a mosaic with boxes on it
            it = 0
            while True:
                image, boxes = self.mosaic_f(index)
                # it is entirely possible than we eot 4 empty images ! 
                if len(boxes)>0:
                    break
                it +=1
                if it == 100:
                    print('Having trouble with loading a mosaic...')
                

            if self.transpose_train and random.random() > 0.5:
                image = np.transpose(image, axes=(1,0,2)) # yxc to xyc
                boxes[:, [0,1,2,3]] = boxes[:, [1,0,3,2]] # flip x/y

            if self.multi_scale_training:
                # symmetric size around 1024
                s = random.randint(self.min_im_size, self.max_im_size)
                resized_image = cv2.resize(image, (s,s), 
                                           interpolation=cv2.INTER_LINEAR)
                if self.pad_multi_scale:
                    image = np.zeros((self.max_im_size, self.max_im_size, 3),
                                     dtype=np.float32)
                    h,w = resized_image.shape[:2]
                    image[:h,:w,:] = resized_image
                else:
                    image = resized_image
                boxes *= s/self.mosaic_size

        #### prepare target
        target = self.make_target(boxes, index)

        ### Transform/augment image and target
        if self.transforms:
            image, target = self.make_nonempty_transform(image, target)

        # yxyx for efficientdet
        if self.yxyx and len(target['bbox'])>0:
            target['bbox'][:,[0,1,2,3]] = target['bbox'][:,[1,0,3,2]]  

        return image, target, self.image_ids[index]


    def make_nonempty_transform(self, image, target):
        """Transform image albumentations, assert there is a box in it."""
        counter = 0
        while True:
            # get a transform
            sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['bbox'],
                    'labels': target['cls']
                })
            # reshape in case its only 1 box
            boxes = np.array(sample['bboxes']).reshape(-1,4) 

            if not self.test:  # remove too small boxes after crop
                w = (boxes[:,2] - boxes[:,0])
                h = (boxes[:,3] - boxes[:,1])
                # discard boxes where w or h <10
                boxes = boxes[(w>=self.size_limit) & (h>=self.size_limit)]
                
            # check if it has boxes
            if (len(boxes) > 0) or self.test:
                image = sample['image']
                target['bbox'] = torch.tensor(boxes).float() 
                if len(boxes) > 0:  # next line fails for empty boxes
                    tb = target['bbox']
                    target['area'] = (tb[:,2] - tb[:,0]) * (tb[:,3] - tb[:,1])
                break

            # hmm hmm
            if counter==100:
                print('Having trouble gettin a non empty image tf ...')
                exit(1)
            counter += 1

        return image, target


    def __len__(self) -> int:
        return self.image_ids.shape[0]


    def load_image_and_boxes(self, index, size=1024):
        """Read image and boxes."""
        # read image as RGB for augmentation by Albumentations
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # convert to xyxy for augmentation by Albumentations
        # drop missing values (images with no boxes)
        records = self.df.query(f'image_id == "{image_id}"').dropna()
        boxes = records[['x', 'y', 'w', 'h']].values
        
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # resize images and boxes
        if size != 1024:
            # isotropically resize image by factor
            f = size/max(image.shape[:2])
            image = cv2.resize(image, None, fx=f, fy=f, 
                               interpolation=cv2.INTER_LINEAR)
            # resize bboxes
            boxes *= f
        
        return image, boxes


    def make_target(self, boxes, index):
        """Create target """
        area = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            'image_id': torch.tensor([index]),
            'area' : area,
            'iscrowd' : iscrowd,
            'bbox' : boxes,  # names fo efficientdet
            'cls' : labels,  # names fo efficientdet
        }

        return target


    def load_mosaic_image_and_boxes(self, index, s=1024, 
                                    minfrac=0.25, maxfrac=0.75):
        """Mosaic"""
        self.mosaic_size = s
        # random breakponints
        xc, yc = np.random.randint(s * minfrac, s * maxfrac,(2,))

        # random other 3 sample (could be the same too...)
        indices = [index] + random.sample(range(len(self.image_ids)), 3) 

        mosaic_image = np.zeros((s, s, 3), dtype=np.float32)
        final_boxes = []

        for i, index in enumerate(indices):

            image, boxes = self.load_image_and_boxes(index, size=s)
            if i == 0:    # top left
                x1a, y1a, x2a, y2a =  0,  0, xc, yc
                x1b, y1b, x2b, y2b = s - xc, s - yc, s, s # from bottom right
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, 0, s , yc
                x1b, y1b, x2b, y2b = 0, s - yc, s - xc, s # from bottom left
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = 0, yc, xc, s
                x1b, y1b, x2b, y2b = s - xc, 0, s, s-yc   # from top right
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc,  s, s
                x1b, y1b, x2b, y2b = 0, 0, s-xc, s-yc    # from top left

            # calculate and apply box offsets due to replacement            
            offset_x = x1a - x1b
            offset_y = y1a - y1b
            boxes[:, 0] += offset_x
            boxes[:, 1] += offset_y
            boxes[:, 2] += offset_x
            boxes[:, 3] += offset_y

            # cut image, save boxes
            mosaic_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            final_boxes.append(boxes)

        # collect boxes
        final_boxes = np.vstack(final_boxes)

        # clip boxes to the image area
        final_boxes[:, 0:] = np.clip(final_boxes[:, 0:], 0, s).astype(np.int32)

        w = (final_boxes[:,2] - final_boxes[:,0])
        h = (final_boxes[:,3] - final_boxes[:,1])
        # discard boxes with no overlap with the final image ( 0 area)
        # final_boxes = final_boxes[w*h>0]

        # discard boxes where w or h <10
        final_boxes = final_boxes[(w>=self.size_limit) & (h>=self.size_limit)]

        return mosaic_image, final_boxes


    def load_stretched_mosaic_image_and_boxes(self, index, s=1024, margin=256):
        """Mosaic, with no complete overlap for larger image."""
        # random breakponints
        s2 = s + margin
        self.mosaic_size = s2
        xc, yc = np.random.randint(margin, s,(2,))

        # random other 3 sample (could be the same too...)
        indices = [index] + random.sample(range(len(self.image_ids)), 3) 

        mosaic_image = np.zeros((s2, s2, 3), dtype=np.float32)
        final_boxes = []

        for i, index in enumerate(indices):
            image, boxes = self.load_image_and_boxes(index, size=s)
            if i == 0:    # top left
                x1a, y1a, x2a, y2a = 0,  0, xc, yc
                x1b, y1b, x2b, y2b = s - xc, s - yc, s, s # from bottom right
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, 0, s2 , yc
                x1b, y1b, x2b, y2b = 0, s - yc, s2 - xc, s # from bottom left
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = 0, yc, xc, s2
                x1b, y1b, x2b, y2b = s - xc, 0, s, s2-yc   # from top right
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc,  s2, s2
                x1b, y1b, x2b, y2b = 0, 0, s2-xc, s2-yc    # from top left

            # calculate and apply box offsets due to replacement            
            offset_x = x1a - x1b
            offset_y = y1a - y1b
            boxes[:, 0] += offset_x
            boxes[:, 1] += offset_y
            boxes[:, 2] += offset_x
            boxes[:, 3] += offset_y

            # cut image, save boxes
            mosaic_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            final_boxes.append(boxes)

        # collect boxes
        final_boxes = np.vstack(final_boxes)

        # clip boxes to the image area
        final_boxes[:, 0:] = np.clip(final_boxes[:, 0:], 0, s2).astype(np.int32)

        # discard boxes with no overlap with the final image ( 0 area)
        w = (final_boxes[:,2] - final_boxes[:,0])
        h = (final_boxes[:,3] - final_boxes[:,1])
        final_boxes = final_boxes[w*h>0]

        return mosaic_image, final_boxes



def collate_fn(batch):
    return tuple(zip(*batch))



def get_train_transforms(train_image_size, random_sized_crop=True, 
                         cutout=False, min_visibility=0.25):
    if random_sized_crop:
        crop_size = (512, 1024)  
    else:
        crop_size = (train_image_size, train_image_size)  

    transforms = [
        A.RandomSizedCrop(min_max_height = crop_size, height = train_image_size, 
                          width = train_image_size, p=1.0),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, 
                                    val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                       p=0.9),
        ],p=0.9),
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ]

    if cutout:
        transforms.append(A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, 
                                   fill_value=0., p=0.5))
        
    transforms.append(ToTensorV2(p=1.0))
        
    return A.Compose(
        transforms,
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=min_visibility,
            label_fields=['labels']
        )
    )


def get_val_transforms(test_image_size):
    return A.Compose(
        [
            A.Resize(height=test_image_size, width=test_image_size, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_data_4_train(config, dir_input='../input/global-wheat-detection',
                     n_splits=5):
    train_df, val_df = next(kfold_by_image(load_data(dir_input),dir_input,
                                          n_splits=n_splits))

    train_dataset = WheatDataset(
        train_df,
        min_im_size =  config.image_size,
        max_im_size =  config.max_image_size,
        transforms=get_train_transforms(config.image_size,
                                        config.random_sized_crop,
                                        config.cutout,
                                        config.min_visibility),
        image_dir=f'{DIR_INPUT}/train',
        multi_scale_training =  config.multi_scale_training,
        use_streched_mosaic = config.use_streched_mosaic,
        size_limit = config.box_size_limit
    )
    val_dataset = WheatDataset(
        val_df,
        transforms=get_val_transforms(config.image_size),
        image_dir=f'{DIR_INPUT}/train', 
        test=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_dataset, val_dataset



def get_data_4_pred(image_size=1024, batch_size=4, num_workers=4,
                    dir_input='../input/global-wheat-detection',
                    n_splits=5):
    _, val_df = next(kfold_by_image(load_data(dir_input),dir_input,
                                    n_splits = n_splits))
    test_df =  make_test_df(f'{DIR_INPUT}/test')

    val_dataset = WheatDataset(
        val_df,
        transforms=get_val_transforms(image_size),
        image_dir=f'{DIR_INPUT}/train', 
        test=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    test_dataset = WheatDataset(
        test_df,
        image_dir = f'{DIR_INPUT}/test',
        test=True,
        transforms=get_val_transforms(image_size)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        sampler=SequentialSampler(test_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )
    
    return test_loader, val_loader, test_dataset, val_dataset



################################################################################
# model_utils.py
################################################################################

import sys
from effdet import  get_efficientdet_config, EfficientDet, DetBenchPredict, DetBenchTrain
from effdet.efficientdet import HeadNet
import gc 

def get_effdet(config_name = 'tf_efficientdet_d5', 
               coco_weights ='data/efficientdet_d5-ef44aea8.pth',
               finetuned_weights = None, image_size=1024):
    """Load a Coco trained or fine tuned efficientdet in train bench.""" 
    config = get_efficientdet_config(config_name)
    net = EfficientDet(config, pretrained_backbone=False)

    if coco_weights:
        checkpoint = torch.load(coco_weights)
        net.load_state_dict(checkpoint)
        del checkpoint
        gc.collect()
        
    config.num_classes = 1
    config.image_size = image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, 
                            norm_kwargs=dict(eps=.001, momentum=.01))
    
    if finetuned_weights:
        checkpoint = torch.load(finetuned_weights)
        # handle the case when I forgot to unwrap the bench ...
        if 'backbone.conv_stem.weight' not in checkpoint['model_state_dict']:
            model_state_dict = {}
            for k,v in checkpoint['model_state_dict'].items():
                if 'model.' == k[:6]:
                    k = k[6:]
                    model_state_dict[k] = v
        else:
            model_state_dict = checkpoint['model_state_dict']
        net.load_state_dict(model_state_dict)

        del checkpoint
        gc.collect()
        

    return DetBenchTrain(net, config).cuda()



################################################################################
# pred_utils.py
################################################################################

import torch
from tqdm import tqdm
from functools import partial
import pandas as pd
import sys

from ensemble_boxes import *
import torch.nn.functional as F


def make_ensemble_test_predictions(config_names, weights, dir_input, tta=True,
                                   verbose=True, n_splits=5, batch_size=8,
                                   image_size=1024):
    """Predict with multiple models, ensemble with WBF, scan thresholds."""
    ### predict with all models, also predict validation data
    boxes_all, scores_all = [], []
    boxes_val_all, scores_val_all = [],[]
    for confname, w in zip(config_names, weights):
        boxes, scores, image_ids, val_preds = make_test_predictions(
            confname, w, dir_input, tta=tta, scan_thresholds=False, conf_th=0.1,
            verbose=verbose, n_splits=n_splits, make_sub=False, 
            batch_size=batch_size, image_size=image_size)
        
        boxes_all.append(boxes)
        scores_all.append(scores)
        boxes_val_all.append(val_preds[0])
        scores_val_all.append(val_preds[1])
        gt_boxes_val = val_preds[2]

    # merge boxes with WBF, test and validation too
    boxes_ensemble, scores_ensemble = merge_boxes_w_wbf(boxes_all, scores_all)
    boxes_val_ensemble, scores_val_ensemble = merge_boxes_w_wbf(
        boxes_val_all, scores_val_all)

    # select best threshold on validation
    boxes_val_ensemble, scores_val_ensemble, best_th = select_best_threshold(
        boxes_val_ensemble, scores_val_ensemble, gt_boxes_val)
    # eval for sanity check during dev
    evaluate(boxes_val_ensemble, scores_val_ensemble, gt_boxes_val, verbose)

    # filter test boxes with this threshold
    boxes_ensemble = [boxes_ensemble[i][scores_ensemble[i]>best_th] 
                      for i in range(len(boxes_ensemble))]
    scores_ensemble = [scores_ensemble[i][scores_ensemble[i]>best_th] 
                       for i in range(len(boxes_ensemble))]

    sub = make_formatted_predictions(boxes_ensemble, scores_ensemble, image_ids,
                                     dir_input)
    sub.to_csv('submission.csv', index=False)


def make_test_predictions(effdet_config_name, w, dir_input, conf_th = 0.45,
                          scan_thresholds=True, tta=True, image_size=1024,
                          batch_size = 2, verbose=True, make_sub=True,
                          n_splits=5):
    """Test prediction pipeline with best threshold selection in validation."""
    # loaders
    test_loader, val_loader,_,_ = get_data_4_pred(
        dir_input=dir_input, image_size=image_size, batch_size=batch_size,
        n_splits=n_splits)
    
    # load model
    net = get_effdet(effdet_config_name, coco_weights = None, 
                     finetuned_weights = w, image_size=image_size)

    # predict validation
    val_boxes, val_scores, val_gt_boxes,_ = predict_whole_dataset(
        net, predict_effdet, val_loader, image_size, tta=tta, 
        conf_th=0.1, verbose=verbose)
    # get best threshold for validation
    boxes_best, scores_best, best_th = select_best_threshold(
        val_boxes, val_scores,  val_gt_boxes)
    # also eval as sanity check
    evaluate(boxes_best, scores_best, val_gt_boxes, verbose)

    # set best threshold value if asked to
    th = best_th if scan_thresholds else conf_th

    # make final predictions on test
    boxes, scores, _, image_ids = predict_whole_dataset(
        net, predict_effdet, test_loader, image_size, tta=tta, conf_th=th,
        verbose=verbose)

    # make formatted submission.csv file
    if make_sub:
        sub = make_formatted_predictions(boxes, scores, image_ids, dir_input)
        sub.to_csv('submission.csv', index=False)

    return boxes, scores, image_ids, (val_boxes, val_scores, val_gt_boxes)


def predict_whole_dataset(model, predict, data_loader, image_size, tta=True,
                          conf_th = 0.5, verbose=True):
    """Make predictions on a whole dataset loader."""
    model.eval()
    if verbose:
        # tqdm prints to new line after interrupting, this makes it more robust
        sys.stdout.flush()
        pbar = tqdm(total=len(data_loader), position=0, leave=True)

    boxes, scores, gt_boxes, all_image_ids = [], [], [], []
    for step, (images, targets, image_ids) in enumerate(data_loader):
        boxes_batch, scores_batch = predict(model, images, image_size, targets,
                                            tta, conf_th)
        boxes += boxes_batch
        scores += scores_batch
        gt_boxes += [t['bbox'].numpy() for t in targets] # dummy in test DS
        all_image_ids += image_ids

        if verbose and (step+1)%10 == 0:
            pbar.update(10)

    if verbose: 
        pbar.close()
        sys.stdout.flush()
    
    return boxes, scores, gt_boxes, all_image_ids


def predict_effdet(model, images, image_size, targets=None, tta=True, 
                   conf_th=0.22):
    """Predict a batch of images with efficientdet."""
    assert isinstance(model, DetBenchTrain) # DetBenchPred has bugs with 1024x?
    images = torch.stack(images).cuda().float()

    with torch.no_grad():
        target = prepare_target(images, targets, image_size) # effdet needs it

        aug_switch = [0,1] if tta else [0] # switches for augmentations  
        sizes = [896, 960, 1024] if tta else [image_size]  # TTA sizes   
        dets = [[] for i in range(images.shape[0])] # empty initial detections

        # loop over possible augmentations
        for s in sizes:
            for lrflip in aug_switch:
                for udflip in aug_switch:
                    for transpose in aug_switch:

                        # augment images and detect boxes on them
                        x = augment_images(images, lrflip, udflip, transpose, s,
                                           image_size)
                        det = model(x, target)['detections']

                        # process detections and them to a list for each image                
                        for i in range(images.shape[0]):
                            di = det[i].detach().cpu().numpy()
                            di = revert_coords(di, lrflip, udflip, transpose, s,
                                               image_size)
                            dets[i].append(di)
                        
        # merge detections from TTA with WBF and format them for final use
        boxes, scores = finalize_detections(dets, conf_th)
    
    return boxes, scores 


def prepare_target(images, targets, image_size):
    """Prepare a target dict required by this efficientdet implementation."""
    # image scale and image size needed for some reason
    target = {}
    target['img_scale'] = torch.tensor([[1.0] 
                                        for _ in images]).cuda().float()
    target['img_size'] = torch.tensor([[image_size, image_size] 
                                        for _ in images]).cuda().float()
    
    # need to add the bbox, cls when is train/eval mode
    fake_bbox = torch.tensor([[42,42,42,42]])
    target['bbox'] = [fake_bbox.cuda().float() for t in targets]
    target['cls'] = [torch.tensor([1]).cuda().float() for t in targets]

    return target


def augment_images(images, lrflip, udflip, transpose, s, image_size):
    """Augment images with 8 flip and rotation combos."""
    # resize & pad with imagenet mean value
    x = F.interpolate(images, size=(s,s), mode='bilinear', align_corners=False)
    x = F.pad(x, (0, image_size - s, 0, image_size - s), value=0.447) 
    
    if lrflip:
        # note the yxyx order used in this efficientdet implementation
        x = x.flip(2)
    if udflip:
        # note the yxyx order used in this efficientdet implementation
        x = x.flip(3)
    if transpose:
        x = torch.transpose(x, 2, 3)
    
    return x


def finalize_detections(dets, th):
    """Merge and format multiple ddetections from augmented images."""
    boxes_batch, scores_batch = [], []
    
    for i in range(len(dets)): # loop over images
        dets[i] = np.vstack(dets[i])   # stack detections from augmentation
        boxes, scores  = dets[i][:,:4], dets[i][:,4]    

        # limit to th
        keep = np.where(scores > th)[0]
        boxes, scores = boxes[keep], scores[keep]
        
        # xyxy to xyhw
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

        if len(boxes)>0:
            boxes, scores = run_wbf(boxes, scores)

        # xyxy to yxyx for consistency
        boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]] 
        
        boxes_batch.append(boxes)
        scores_batch.append(scores)

    return boxes_batch, scores_batch


def revert_coords(di, lrflip, udflip, transpose, s, image_size):
    """Revert prediction coordinates after augmentation."""
    # NOTE Transposition and flipping DOES NOT COMMUTE
    # so it must come in reverse order as applied 
    if transpose: 
        di[:,[0,1,2,3]] = di[:,[1,0,3,2]]
    if lrflip: 
        # note the yxyx order used in this efficientdet implementation
        di[:,1] = image_size - di[:,1] - di[:,3]    
    if udflip: 
        # note the yxyx order used in this efficientdet implementation
        di[:,0] = image_size - di[:,0] - di[:,2]
    
    # resize boxes to original size
    di[:,:4] *= (image_size / s )
    
    return di




def run_wbf(boxes, scores, iou_thr=0.5, skip_box_thr=0.34, weights=None):
    """ Run Weighted boxes fusion on predictions."""
    # sort them to make sure
    boxes = boxes[np.argsort(scores)[::-1]]
    scores = np.sort(scores)[::-1]

    # make fake labels for single cls dataset
    labels0 = np.ones(len(scores))

    # make boxes 0-1
    maxbox = np.max(boxes)
    boxes /= maxbox

    # run wbf on a single image 
    boxes, scores, labels = weighted_boxes_fusion(
        [boxes], [scores], [labels0], weights=None, iou_thr=iou_thr, 
        skip_box_thr=skip_box_thr)
    
    boxes *= maxbox  # scale back boxes

    return boxes, scores


def make_formatted_predictions(boxes, scores, image_ids, dir_input):
    """Make predictions and format them to GWD competition format."""
    pred_dict = {'image_id':[],'PredictionString':[]}
    for i in range(len(image_ids)):
        path = f'{dir_input}/test/{image_ids[i]}.jpg'
        im0shape = cv2.imread(path).shape[:2]

        preds=''
        for box,score in zip(boxes[i],scores[i]):
            box[[0,2]] *= im0shape[0]/1024.
            box[[1,3]] *= im0shape[1]/1024.
            position_ints = ' '.join(map(str,int_yxyx2xywh(box)))
            preds += ' %.4f ' % score + position_ints

        pred_dict['image_id'].append(image_ids[i])
        pred_dict['PredictionString'].append(preds.strip())

    return pd.DataFrame(pred_dict)
    
    

def merge_boxes_w_wbf(boxes, scores):
    """Merge boxes from multiple predictions with WBF."""
    merged_boxes, merged_scores = [], []
    for i in range(len(boxes[0])):
        b = np.vstack([boxes[j][i] for j in range(len(boxes))])
        s = np.concatenate([scores[j][i] for j in range(len(boxes))])

        b = b[np.argsort(s)[::-1]]
        s = np.sort(s)[::-1]

        if len(b) > 0:
            b, s = run_wbf(b, s)
        
        merged_boxes.append(b)
        merged_scores.append(s)

    return merged_boxes, merged_scores


def int_yxyx2xywh(coords):
    x = int(round(coords[1]))
    y = int(round(coords[0]))
    w = int(round(coords[3]-coords[1]))
    h = int(round(coords[2]-coords[0]))
    return x,y,w,h




################################################################################
# eval.py
################################################################################


import numba
from numba import jit
import numpy as np


def evaluate(boxes, scores, gt_boxes, verbose=True,
             iou_ths = tuple(np.linspace(0.5, 0.75, 6))):
    """Evaluate a predictions with GWD competition metric."""
    APs, AP50s, AP75s  = [], [], []
    for i in range(len(boxes)):
        # check for empty GT images:
        if len(gt_boxes[i])==0:  # if there are any boxes -> 0
            if len(boxes[i])>0:
                APs.append(0)
                AP50s.append(0)
                AP75s.append(0)
            # no prexdicted boxes, lucky skip
            continue

        # sort to be on the safe side, although they should come sorted
        boxes_sorted = boxes[i][np.argsort(scores[i])[::-1]]
        image_precisions = calculate_AP(boxes_sorted,
                                        gt_boxes[i],
                                        ths = iou_ths,
                                        form='pascal_voc')
        APs.append(image_precisions[0])
        AP50s.append(image_precisions[1])
        AP75s.append(image_precisions[2])

    if verbose:
        print("mAP: {0:.5f}".format(np.mean(APs)), end=', ')
        print("AP@.5: {0:.5f}".format(np.mean(AP50s)), end=', ')
        print("AP@.75: {0:.5f}".format(np.mean(AP75s)))

    return np.mean(APs)


def select_best_threshold(boxes, scores, gt_boxes, verbose=False):
    # https://www.kaggle.com/hawkey/yolov5-pseudo-labeling-oof-evaluation
    best_th = 0
    best_mAP = 0
    for th in np.arange(0.01, 1, 0.01):
        if verbose:
            print('TH: %.2f'%th, end=', ')
        boxes_after_th = [boxes[i][scores[i]>th] for i in range(len(boxes))]
        scores_after_th = [scores[i][scores[i]>th] for i in range(len(boxes))]
        mAP = evaluate(boxes_after_th, scores_after_th, gt_boxes, verbose)
        if mAP > best_mAP:
            best_mAP = mAP
            best_th = th

    if verbose:
        print('-'*30)
        print(f'[Best Score Threshold]: {best_th}')
        print(f'[Best mAP]: {best_mAP:.4f}')
        print('-'*30)

    boxes_best = [boxes[i][scores[i]>best_th] for i in range(len(boxes))]
    scores_best = [scores[i][scores[i]>best_th] for i in range(len(boxes))]

    return  boxes_best, scores_best, best_th

@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area




@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, th = 0.5, form = 'pascal_voc', 
                    ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available 
                ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing 
                calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < th:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx
    

@jit(nopython=True)
def get_precision(gts, preds, th = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available 
        ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted 
                boxes,sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated 
                ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    for pred_idx in range(n):
        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            th=th, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box 
            # with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_AP(gt_boxes, preds, ths = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available 
                ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted 
                boxes, sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_th = len(ths)
    AP, AP50, AP75 = 0.0, 0.0, 0.0    
    ious = np.ones((len(gt_boxes), len(preds))) * -1

    for th in ths:
        precision_at_th = get_precision(gt_boxes.copy(), preds, th=th,
                                         form=form, ious=ious)
        AP += precision_at_th / n_th
        if th==0.5:
            AP50 = precision_at_th
        if th==0.75:
            AP75 = precision_at_th

    return AP, AP50, AP75




################################################################################
# train_utils.py
################################################################################

import torch
from timm.utils import ModelEma
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import time
from datetime import datetime
import os
from glob import glob

try:
    from apex import amp
    mixed_precision = True
    print('Mixed precision available')
except ModuleNotFoundError:
    mixed_precision = False


def train(effdet_config_name, w, dir_input, config, n_splits=5):
    """Train effdet."""
    # loaders
    train_loader, val_loader,_,_ = get_data_4_train(config, dir_input=dir_input,
                                                    n_splits=n_splits)

    # load model
    net = get_effdet(effdet_config_name, coco_weights = w, 
                     image_size=config.image_size)

    # fitter class fo convencience
    fitter = Fitter(model=net, config=config)
    fitter.fit(train_loader, val_loader)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:
    
    def __init__(self, model, config):
        self.config = config
        self.epoch = 0

        self.base_dir = config.folder
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_val_mAP = 0

        self.model = model

        # optimizer no decay for bias and batch norm
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape) == 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
        pg = [{'params': no_decay, 'weight_decay': 0.},
              {'params': decay, 'weight_decay': 1e-3}]

        if config.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(pg, lr=config.lr, momentum=0.9,
                                             nesterov=True)
        elif config.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.AdamW(pg, lr=config.lr)

        if self.config.use_amp:
            self.model, optimizer = amp.initialize(self.model, self.optimizer, 
                                                   opt_level='O1', verbosity=0)
            
        self.model2eval = self.model # will change to EMA during 
        self.model_ema = None

        self.scheduler = config.SchedulerClass(self.optimizer, 
                                               **config.scheduler_params)

    def switch_on_EMA(self, val_mAP):
        """Switch on model ema when map is higher than a threshold."""
        if (self.config.use_model_ema and
                val_mAP > self.config.model_ema_limit and
                self.model_ema is None) :
            self.model_ema =  ModelEma(self.model, 
                                       decay=self.config.model_ema_decay)
            self.model2eval = self.model_ema.ema
            self.log('Switched on Model EMA')
            

    def validate(self, validation_loader, dataset='Val'):
        t = time.time()

        # predict boxes
        boxes, scores, gt_boxes, _ = predict_whole_dataset(
            self.model2eval, predict_effdet, validation_loader, 
            self.config.image_size, conf_th = 0.1, tta=False)

        # get best threshold for validation
        boxes_best, scores_best, best_th = select_best_threshold(boxes, scores, 
                                                                 gt_boxes)
        # also eval as sanity check, and print it
        val_mAP = evaluate(boxes_best,scores_best,gt_boxes,self.config.verbose)

        if self.config.verbose:
            name = dataset + ' EMA' if self.model_ema else  dataset
            self.log(f'[RESULT]: {name} Epoch: {self.epoch}, mAP: {val_mAP:.5f}, time: {(time.time() - t):.5f}')
        
        return val_mAP

    def save_best_model(self):
        self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
        best_paths = glob(f'{self.base_dir}/best-checkpoint-*epoch.bin')
        # remove every single previous best checkpoint
        for path in sorted(best_paths)[:-1]:
            # trick: first make filesize 0, GDRIVE trash gets full            
            open(path, 'w').close()
            # remove 
            os.remove(path)

    def fit(self, train_loader, validation_loader):
        if self.config.test_first:  # test first (debug and other reasons)
            val_mAP = self.validate(validation_loader)
            self.switch_on_EMA(val_mAP)   # only when conditions are met 

        for e in range(self.config.n_epochs):
            lr =  self.optimizer.param_groups[0]['lr']
            self.log(f'\nLR:{lr}, {datetime.utcnow().isoformat()}')

            self.train_one_epoch(train_loader)

            val_mAP = self.validate(validation_loader)

            if val_mAP > self.best_val_mAP:
                self.save_best_model()
                self.best_val_mAP = val_mAP
                
            self.switch_on_EMA(val_mAP)   # only when conditions are met 

            if (e+1) % 10 == 0: # evaluate train, but not too often, its slow...
                self.validate(train_loader, 'Train')

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=val_mAP)

            self.epoch += 1

    def train_one_epoch(self, train_loader):
        t = time.time()
        self.model.train()
        summary_loss = AverageMeter()

        if self.config.verbose:
            sys.stdout.flush()
            pbar = tqdm(total=len(train_loader), position=0, leave=True)

        self.optimizer.zero_grad()
        for step, (images, targets, image_ids) in enumerate(train_loader):            
            images = torch.stack(images).cuda().float()
            target = {'bbox': [t['bbox'].cuda().float() for t in targets],
                      'cls': [t['cls'].cuda().float() for t in targets]}
            
            loss = self.model(images, target)['loss']
            loss = loss / self.config.accumulate

            if self.config.use_amp:
              with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step+1) % self.config.accumulate == 0:
                # Gradient clipping if desired:
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
                self.optimizer.step()
                if self.model_ema:
                    self.model_ema.update(self.model)
                self.optimizer.zero_grad()

            if self.config.verbose and step%10 == 0:
                desc = 'mem: %.1fG'%(torch.cuda.memory_cached()/1e9)
                desc += ', Loss: %.3f' % summary_loss.avg
                desc += ', LR: %.2e' % self.optimizer.param_groups[0]['lr']
                pbar.set_description(desc)
                pbar.update(10)

            summary_loss.update(loss.detach().item(), images.shape[0])

            if self.config.step_scheduler:
                self.scheduler.step()
        
        if self.config.verbose:
            pbar.close()
            sys.stdout.flush()
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
        return summary_loss
    
    def save(self, path):
        self.model.eval()
        if self.model_ema:
            model_state_dict = get_state_dict(self.model_ema)
        else:
            model_state_dict = self.model.model.state_dict()

        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_mAP': self.best_val_mAP,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_mAP = checkpoint['best_val_mAP']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


def unwrap_model(model):
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.module if hasattr(model, 'module') else model


def get_state_dict(model):
    return unwrap_model(model).state_dict()


class TrainGlobalConfig:
    folder = 'tmp'
    
    # augmentations
    image_size = 768
    max_image_size = 1792
    multi_scale_training = True
    random_sized_crop = False
    assert not (random_sized_crop and multi_scale_training)   # only one multi scale method!
    cutout = False
    use_streched_mosaic = True
    box_size_limit = 10
    min_visibility = 0.25

    n_epochs = 40
    optimizer = 'adam'
    lr = 0.0001
    num_workers = 4
    batch_size = 4
    accumulate = 1
    val_batch_size = 1
    
    use_amp = True
    use_model_ema = True
    model_ema_decay = 0.9998
    model_ema_limit = 0.68

    # -------------------
    verbose = VERBOSE
    test_first = False

    # --------------------
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation

    SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = dict(
         max_lr=0.00025,
         epochs=n_epochs,
         steps_per_epoch = 684, # BS=4
         pct_start=0.1,
         anneal_strategy='cos', 
         div_factor = 2,
         final_div_factor=4
    )


################################################################################
# main
################################################################################

# seed something (mostly data order)
seed_everything(SEED)

# uncomment to inspect images
#_,_,train_dataset, val_dataset =  get_data_4_train(TrainGlobalConfig, DIR_INPUT)

# uncomment to train
#train(CONFIG_NAME, COCO_W, DIR_INPUT, TrainGlobalConfig, verbose=VERBOSE)

# uncomment to predict with single model
make_test_predictions(CONFIG_NAMES[0],W[0],DIR_INPUT,tta=TTA,verbose=VERBOSE)

# uncomment to predict with multiple models
# make_ensemble_test_predictions(CONFIG_NAMES, W, DIR_INPUT, TTA, VERBOSE)