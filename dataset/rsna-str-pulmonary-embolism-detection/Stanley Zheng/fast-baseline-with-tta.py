'''
Versions list:
v1: Turbo mode added! Extremely fast inference
v2: Add TTA (only flips)
v3: Other TTA (doesn't work), added TTA_STEPS variable to change TTA times
v4: Complete notebook run of TTAx4 to see runtime (limit should be about 8700s)
v5: Patched albumentations bug with ToTensorV2 and division
v6: Roll back TTA in order to test if notebook works fully 
v7: added brightness, contrast, hue augmentation, full run for submit
v8: Same as v7, but further optimization, since maybe v7 cannot be submitted. Not sure if the changes here are actually going to work (move normalizaton after augmentations)
v9: Reverted v8 since its results were not reproducible
'''

package_path = '../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
import sys; sys.path.append(package_path)

bash_commands = [
            'cp ../input/gdcm-conda-install/gdcm.tar .',
            'tar -xvzf gdcm.tar',
            'conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2'
            ]

import subprocess
for bashCommand in bash_commands:
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
import pydicom
from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import albumentations as albu

do_full = True

TTA_STEPS=4

CFG = {
    'train': False,
    
    'train_img_path': '../input/rsna-str-pulmonary-embolism-detection/train',
    'test_img_path': '../input/rsna-str-pulmonary-embolism-detection/test',
    'cv_fold_path': '../input/stratified-validation-strategy/rsna_train_splits_fold_20.csv',
    'train_path': '../input/rsna-str-pulmonary-embolism-detection/train.csv',
    'test_path': '../input/rsna-str-pulmonary-embolism-detection/test.csv',
    
    'image_target_cols': [
        'pe_present_on_image', # only image level
    ],
    
    'exam_target_cols': [
        'negative_exam_for_pe', # exam level
        'rv_lv_ratio_gte_1', # exam level
        'rv_lv_ratio_lt_1', # exam level
        'leftsided_pe', # exam level
        'chronic_pe', # exam level
        'rightsided_pe', # exam level
        'acute_and_chronic_pe', # exam level
        'central_pe', # exam level
        'indeterminate' # exam level
    ], 
    
    'img_num': 200,
    'img_size': 256,
    'lr': 0.0005,
    'epochs': 2,
    'device': 'cuda', # cuda, cpu
    'train_bs': 2,
    'accum_iter': 8,
    'verbose_step': 1,
    'num_workers': 4,
    'efbnet': 'efficientnet-b0',
    
    'train_folds': [np.arange(0,16),
                    np.concatenate([np.arange(0,12), np.arange(16,20)]),
                    np.concatenate([np.arange(0,8), np.arange(12,20)]),
                    np.concatenate([np.arange(0,4), np.arange(8,20)]),
                    np.arange(4,20),
                   ],#[np.arange(0,16)],
    
    'valid_folds': [np.arange(16,20),
                    np.arange(12,16),
                    np.arange(8,12),
                    np.arange(4,8),
                    np.arange(0,4)
                   ],#[np.arange(16,20)],
    
    'model_path': '../input/kh-rsna-model',
    'tag': 'efb0_stage2_multilabel'
}

SEED = 42321

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X-np.min(X) 
    X = X*np.reciprocal(np.max(X))
    return X

def get_img(path, transforms):
    
    d = pydicom.read_file(path)

    '''
    RED channel / LUNG window / level=-600, width=1500
    GREEN channel / PE window / level=100, width=700
    BLUE channel / MEDIASTINAL window / level=40, width=400
    '''

    img = (d.pixel_array * d.RescaleSlope) + d.RescaleIntercept
    
    r = window(img, -600, 1500)
    g = window(img, 100, 700)
    b = window(img, 40, 400)
    
    res = np.concatenate([r[:, :, np.newaxis],
                          g[:, :, np.newaxis],
                          b[:, :, np.newaxis]], axis=-1)
    
    res = transforms(image=res)['image']

    return res

def tta_augmentation():
    transforms = [
        albu.FromFloat(p=1., dtype=np.uint8, max_value=255.),# required to use cv2 computations
        albu.Resize(256, 256), # Try contrast, noise, artefacts, cutout
        albu.HorizontalFlip(p=0.25),
        albu.VerticalFlip(p=0.25),
        albu.RandomRotate90(p=0.25),
        albu.RandomBrightnessContrast(p=0.5),
        albu.HueSaturationValue(p=0.5),
        albu.ToFloat(p=1.),
        ToTensorV2(p=1.),
    ]
    return albu.Compose(transforms)

def get_valid_transforms():
    return albu.Compose([
            albu.Resize(256, 256),
            ToTensorV2(p=1.),
        ], p=1.)

class RSNADatasetStage1(Dataset):
    def __init__(
        self, df, label_smoothing, data_root, 
        image_subsampling=True, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.label_smoothing = label_smoothing
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.df[CFG['image_target_cols']].values[index]
            
        path = "{}/{}/{}/{}.dcm".format(self.data_root, 
                                        self.df.iloc[index]['StudyInstanceUID'], 
                                        self.df.iloc[index]['SeriesInstanceUID'], 
                                        self.df.iloc[index]['SOPInstanceUID'])

        img  = get_img(path, self.transforms)

        return img
        
class RSNADataset(Dataset):
    def __init__(
        self, df, label_smoothing, data_root, 
        image_subsampling=True, transforms=None, output_label=True
    ):

        super().__init__()
        self.df = df
        self.patients = self.df['StudyInstanceUID'].unique()
        self.image_subsampling = image_subsampling
        self.label_smoothing = label_smoothing
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        
    def get_patients(self):
        return self.patients
        
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index: int):
        
        patient = self.patients[index]
        df_ = self.df.loc[self.df.StudyInstanceUID == patient]
        
        per_image_feats = get_stage1_columns()
        #print(per_image_feats)
        
        if self.image_subsampling:
            img_num = min(CFG['img_num'], df_.shape[0])
            
            # naive image subsampling
            img_ix = np.random.choice(np.arange(df_.shape[0]), replace=False, size=img_num)
            
            # get all images, then slice location and sort according to z values
            imgs = np.zeros((CFG['img_num'],), np.float32) #np.zeros((CFG['img_num'], CFG['img_size'], CFG['img_size'], 3), np.float32)
            per_image_preds = np.zeros((CFG['img_num'], len(per_image_feats)), np.float32)
            locs = np.zeros((CFG['img_num'],), np.float32)
            image_masks = np.zeros((CFG['img_num'],), np.float32)
            image_masks[:img_num] = 1.
            
            # get labels
            if self.output_label:
                exam_label = df_[CFG['exam_target_cols']].values[0]
                image_labels = np.zeros((CFG['img_num'], len(CFG['image_target_cols'])), np.float32)
            
        else:
            img_num = df_.shape[0]
            img_ix = np.arange(df_.shape[0])
            
            # get all images, then slice location and sort according to z values
            imgs = np.zeros((img_num, ), np.float32) #np.zeros((img_num, CFG['img_size'], CFG['img_size'], 3), np.float32)
            per_image_preds = np.zeros((img_num, len(per_image_feats)), np.float32)
            locs = np.zeros((img_num,), np.float32)
            image_masks = np.zeros((img_num,), np.float32)
            image_masks[:img_num] = 1.

            # get labels
            if self.output_label:
                exam_label = df_[CFG['exam_target_cols']].values[0]
                image_labels = np.zeros((img_num, len(CFG['image_target_cols'])), np.float32)
                
        for i, im_ix in enumerate(img_ix):
            path = "{}/{}/{}/{}.dcm".format(self.data_root, 
                                            df_['StudyInstanceUID'].values[im_ix], 
                                            df_['SeriesInstanceUID'].values[im_ix], 
                                            df_['SOPInstanceUID'].values[im_ix])
            
            d = pydicom.read_file(path)
            locs[i] = d.ImagePositionPatient[2]
            per_image_preds[i,:] = df_[per_image_feats].values[im_ix,:]
            
            if self.output_label == True:
                image_labels[i] = df_[CFG['image_target_cols']].values[im_ix]

        #print('get img done')
        
        seq_ix = np.argsort(locs)
        
        # image features: img_num * img_size * img_size * 1
        '''
        imgs = imgs[seq_ix]
        if self.transforms:
            imgs = [self.transforms(image=img)['image'] for img in imgs]
        imgs = torch.stack(imgs)
        '''
        
        # image level features: img_num
        #locs[:img_num] -= locs[:img_num].min()
        locs = locs[seq_ix]
        locs[1:img_num] = locs[1:img_num]-locs[0:img_num-1]
        locs[0] = 0
        
        per_image_preds = per_image_preds[seq_ix]
        
        # patient level features: 1
        
        # train, train-time valid, multiple patients: imgs, locs, image_labels, exam_label, img_num
        # whole valid-time valid, single patient: imgs, locs, image_labels, exam_label, img_num, sorted id
        # whole test-time test, single patient: imgs, locs, img_num, sorted_id
        
        # do label smoothing
        if self.output_label == True:
            image_labels = image_labels[seq_ix]
            image_labels = np.clip(image_labels, self.label_smoothing, 1 - self.label_smoothing)
            exam_label =  np.clip(exam_label, self.label_smoothing, 1 - self.label_smoothing)
            
            return imgs, per_image_preds, locs, image_labels, exam_label, image_masks
        else:
            return imgs, per_image_preds, locs, img_num, index, seq_ix

from albumentations.pytorch import ToTensorV2
        
class RNSAImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = EfficientNet.from_name(CFG['efbnet'])
        #print(self.cnn_model, CFG['efbnet'])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
    def get_dim(self):
        return self.cnn_model._fc.in_features
        
    def forward(self, x):
        feats = self.cnn_model.extract_features(x)
        return self.pooling(feats).view(x.shape[0], -1)   

class RSNAImgClassifierSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = RNSAImageFeatureExtractor()
        self.image_predictors = nn.Linear(self.cnn_model.get_dim(), 1)
        
    def forward(self, imgs):
        #print(images.shape)
        imgs_embdes = self.cnn_model(imgs) # bs * efb_feat_size
        #print(imgs_embdes.shape)
        image_preds = self.image_predictors(imgs_embdes)
        
        return image_preds

class RSNAImgClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = RNSAImageFeatureExtractor()
        self.image_predictors = nn.Linear(self.cnn_model.get_dim(), 9)
        
    def forward(self, imgs):
        #print(images.shape)
        imgs_embdes = self.cnn_model(imgs) # bs * efb_feat_size
        #print(imgs_embdes.shape)
        image_preds = self.image_predictors(imgs_embdes)
        
        return image_preds
    
def post_process(exam_pred, image_pred):
    
    rv_lv_ratio_lt_1_ix = CFG['exam_target_cols'].index('rv_lv_ratio_lt_1')
    rv_lv_ratio_gte_1_ix = CFG['exam_target_cols'].index('rv_lv_ratio_gte_1')
    central_pe_ix = CFG['exam_target_cols'].index('central_pe')
    rightsided_pe_ix = CFG['exam_target_cols'].index('rightsided_pe')
    leftsided_pe_ix = CFG['exam_target_cols'].index('leftsided_pe')
    acute_and_chronic_pe_ix = CFG['exam_target_cols'].index('acute_and_chronic_pe')
    chronic_pe_ix = CFG['exam_target_cols'].index('chronic_pe')
    negative_exam_for_pe_ix = CFG['exam_target_cols'].index('negative_exam_for_pe')
    indeterminate_ix = CFG['exam_target_cols'].index('indeterminate')
    
    # rule 1 or rule 2 judgement: if any pe image exist
    has_pe_image = torch.max(image_pred, 1)[0][0] > 0
    #print(has_pe_image)
    
    # rule 1-a: only one >= 0.5, the other < 0.5
    rv_lv_ratios = exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix]]
    rv_lv_ratios_1_a = nn.functional.softmax(rv_lv_ratios, dim=1) # to make one at least > 0.5
    rv_lv_ratios_1_a = torch.log(rv_lv_ratios_1_a/(1-rv_lv_ratios_1_a)) # turn back into logits
    exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix]] = torch.where(has_pe_image, rv_lv_ratios_1_a, rv_lv_ratios)
    
    # rule 1-b-1 or 1-b-2 judgement: at least one > 0.5
    crl_pe = exam_pred[:, [central_pe_ix, rightsided_pe_ix, leftsided_pe_ix]]
    has_no_pe = torch.max(crl_pe ,1)[0] <= 0 # all <= 0.5
    #print(has_no_pe)
    #assert False
        
    # rule 1-b
    max_val = torch.max(crl_pe, 1)[0]
    crl_pe_1_b = torch.where(crl_pe==max_val, 0.0001-crl_pe+crl_pe, crl_pe)
    exam_pred[:, [central_pe_ix, rightsided_pe_ix, leftsided_pe_ix]] = torch.where(has_pe_image*has_no_pe, crl_pe_1_b, crl_pe)
    
    # rule 1-c-1 or 1-c-2 judgement: at most one > 0.5
    ac_pe = exam_pred[:, [acute_and_chronic_pe_ix, chronic_pe_ix]]
    both_ac_ch = torch.min(ac_pe ,1)[0] > 0 # all > 0.5
    
    # rule 1-c
    ac_pe_1_c = nn.functional.softmax(ac_pe, dim=1) # to make only one > 0.5
    ac_pe_1_c = torch.log(ac_pe_1_c/(1-ac_pe_1_c)) # turn back into logits
    exam_pred[:, [acute_and_chronic_pe_ix, chronic_pe_ix]] = torch.where(has_pe_image*both_ac_ch, ac_pe_1_c, ac_pe)
    
    # rule 1-d
    neg_ind = exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]]
    neg_ind_1d = torch.clamp(neg_ind, max=0)
    exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]] = torch.where(has_pe_image, neg_ind_1d, neg_ind)
    
    # rule 2-a
    ne_inde = exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]]
    ne_inde_2_a = nn.functional.softmax(ne_inde, dim=1) # to make one at least > 0.5
    ne_inde_2_a = torch.log(ne_inde_2_a/(1-ne_inde_2_a)) # turn back into logits
    exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]] = torch.where(~has_pe_image, ne_inde_2_a, ne_inde)
    
    # rule 2-b
    all_other_exam_labels = exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix,
                                          central_pe_ix, rightsided_pe_ix, leftsided_pe_ix,
                                          acute_and_chronic_pe_ix, chronic_pe_ix]]
    all_other_exam_labels_2_b = torch.clamp(all_other_exam_labels, max=0)
    exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix,
                  central_pe_ix, rightsided_pe_ix, leftsided_pe_ix,
                  acute_and_chronic_pe_ix, chronic_pe_ix]] = torch.where(~has_pe_image, all_other_exam_labels_2_b, all_other_exam_labels)
    
    return exam_pred, image_pred
    
def check_label_consistency(checking_df):
    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS
    df = checking_df.copy()
    print(df.shape)
    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())

    df_pos = df.loc[df.positive_images_in_exam >  0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]

    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 
                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)
    rule1a['broken_rule'] = '1a'

    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 
                        (df_pos.rightsided_pe <= 0.5) & 
                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)
    rule1b['broken_rule'] = '1b'

    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 
                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)
    rule1c['broken_rule'] = '1c'
    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS

    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 
                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)
    rule1d['broken_rule'] = '1d'

    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 
                         (df_neg.negative_exam_for_pe >  0.5)) | 
                        ((df_neg.indeterminate        <= 0.5)  & 
                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)
    rule2a['broken_rule'] = '2a'

    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 
                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |
                        (df_neg.central_pe           > 0.5) | 
                        (df_neg.rightsided_pe        > 0.5) | 
                        (df_neg.leftsided_pe         > 0.5) |
                        (df_neg.acute_and_chronic_pe > 0.5) | 
                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)
    rule2b['broken_rule'] = '2b'
    # MERGING INCONSISTENT PREDICTIONS
    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis = 0)
    
    print('label in-consistency counts:', errors.shape)
        
    if errors.shape[0] > 0:
        print(errors.broken_rule.value_counts())
        print(errors)
        assert False

class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        x_size= x.size()
        c_in = x.contiguous().view(x_size[0] * x_size[1], *x_size[2:])
        
        c_out = self.module(c_in)
        r_in = c_out.view(x_size[0], x_size[1], -1)
        if self.batch_first is False:
            r_in = r_in.permute(1, 0, 2)
        return r_in 

def inference(model, device, df, root_path):
    model.eval()

    t = time.time()

    ds = RSNADataset(df, 0.0, root_path,  image_subsampling=False, transforms=tta_augmentation(), output_label=False) # change transforms=get_valiid_augmentation() to avoid TTA, or tta_augmentation()
    
    dataloader = torch.utils.data.DataLoader(
        ds, 
        batch_size=1,
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    
    patients = ds.get_patients()
    
    res_dfs = []

    for step, (imgs, per_image_preds, locs, img_num, index, seq_ix) in enumerate(dataloader):
        imgs = imgs.to(device).float()
        per_image_preds = per_image_preds.to(device).float()
        locs = locs.to(device).float()
        
        index = index.detach().numpy()[0]
        seq_ix = seq_ix.detach().numpy()[0,:]
        
        patient_filt = (df.StudyInstanceUID == patients[index])
        
        patient_df = pd.DataFrame()
        patient_df['SOPInstanceUID'] = df.loc[patient_filt, 'SOPInstanceUID'].values[seq_ix]
        patient_df['SeriesInstanceUID'] = df.loc[patient_filt, 'SeriesInstanceUID'].values # no need to sort
        patient_df['StudyInstanceUID'] = patients[index] # single value
        
        for c in CFG['image_target_cols']+CFG['exam_target_cols']:
            patient_df[c] = 0.0

        #with autocast():
        image_preds, exam_pred = model(per_image_preds, locs)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)
        
        exam_pred, image_preds = post_process(exam_pred, image_preds)
        
        exam_pred = torch.sigmoid(exam_pred).cpu().detach().numpy()
        image_preds = torch.sigmoid(image_preds).cpu().detach().numpy()

        patient_df[CFG['exam_target_cols']] = exam_pred[0]
        patient_df[CFG['image_target_cols']] = image_preds[0,:]
        res_dfs += [patient_df]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(dataloader)):
            print(
                f'Inference Step {step+1}/{len(dataloader)}, ' + \
                f'time: {(time.time() - t):.4f}', end='\r' if (step + 1) != len(dataloader) else '\n'
            )
    
    res_dfs = pd.concat(res_dfs, axis=0).reset_index(drop=True)
    res_dfs = df[['SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID']].merge(res_dfs, on=['SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID'], how='left')
    print(res_dfs[CFG['image_target_cols']+CFG['exam_target_cols']].head(5))
    print(res_dfs[CFG['image_target_cols']+CFG['exam_target_cols']].tail(5))
    assert res_dfs.shape[0] == df.shape[0]
    check_label_consistency(res_dfs)
    
    return res_dfs
  
STAGE1_CFGS = [
    {
        'tag': 'efb0_stage1',
        'model_constructor': RSNAImgClassifierSingle,
        'dataset_constructor': RSNADatasetStage1,
        'output_len': 1
    },
    {
        'tag': 'efb0_stage1_multilabel',
        'model_constructor': RSNAImgClassifier,
        'dataset_constructor': RSNADatasetStage1,
        'output_len': 9
    },
]
STAGE1_CFGS_TAG = 'efb0-stage1-single-multi-label'


def get_stage1_columns():
    
    new_feats = []
    for cfg in STAGE1_CFGS: # CHECK THIS OUT, DOES IT WORK
        for i in range(cfg['output_len']):
            f = cfg['tag']+'_'+str(i)
            new_feats += [f]
        
    return new_feats

def update_stage1_test_preds(df):
    
    new_feats = get_stage1_columns()
    
    for f in new_feats:
        df[f] = 0
    df.loc[:,new_feats] = 0 
    models = []

    for cfg in STAGE1_CFGS:
        device = torch.device(CFG['device'])
        model = cfg['model_constructor']().to(device)
        model.load_state_dict(torch.load('{}/model_{}'.format(CFG['model_path'], cfg['tag'])))
        model.eval()
        models.append(model)
    for n, i in enumerate(range(TTA_STEPS)):
        if n==0:
           test_ds = RSNADatasetStage1(df, 0.0, CFG['test_img_path'],  image_subsampling=False, transforms=get_valid_transforms(), output_label=False) # transforms=get_valid_transforms() or transforms=tta_augmentation()
        else:
            test_ds = RSNADatasetStage1(df, 0.0, CFG['test_img_path'],  image_subsampling=False, transforms=tta_augmentation(), output_label=False)

        test_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=256,
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
            sampler=SequentialSampler(test_ds)
        )
        image_preds_all_list = []

        image_preds_all = []
        for step, imgs in enumerate(tqdm(test_loader)):
            #imgs = torch.reshape(imgs, (-1, 3, 256, 256))
            imgs = imgs.to(device).float()
            for model in models:
                image_preds = model(imgs)   #output = model(input)
                image_preds_all += [image_preds.cpu().detach().numpy()]

        del test_loader

        image_preds_all_image = np.concatenate(image_preds_all[::2], axis=0)
        image_preds_all_exam = np.concatenate(image_preds_all[1::2], axis=0)

        image_preds_all = np.concatenate([image_preds_all_image, image_preds_all_exam], axis=1)

        #image_preds_all = np.concatenate(image_preds_all, axis=1)
        print(np.array(new_feats).shape)
        print(np.array(image_preds_all).shape)
        df.loc[:,new_feats] += image_preds_all

    torch.cuda.empty_cache()
    df.loc[:, new_feats] /= TTA_STEPS

    return df

class RSNAClassifier(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        
        self.gru = nn.GRU(len(get_stage1_columns())+1, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        
        self.image_predictors = TimeDistributed(nn.Linear(hidden_size*2, 1))
        self.exam_predictor = nn.Linear(hidden_size*2*2, 9)
        
    def forward(self, img_preds, locs):
        
        embeds = torch.cat([img_preds, locs.view(locs.shape[0], locs.shape[1], 1)], dim=2) # bs * ts * fs
        
        embeds, _ = self.gru(embeds)
        image_preds = self.image_predictors(embeds)
        
        avg_pool = torch.mean(embeds, 1)
        max_pool, _ = torch.max(embeds, 1)
        conc = torch.cat([avg_pool, max_pool], 1)
        
        exam_pred = self.exam_predictor(conc)
        return image_preds, exam_pred

if __name__ == '__main__':

    seed_everything(SEED)
    from os import path
    if path.exists('../input/rsna-str-pulmonary-embolism-detection/train') and not do_full:
        test_df = pd.read_csv(CFG['test_path']).head(1000)
    else:
        test_df = pd.read_csv(CFG['test_path'])
    
    with torch.no_grad():
        test_df = update_stage1_test_preds(test_df)
    device = torch.device(CFG['device'])
    model = RSNAClassifier().to(device)
    model.load_state_dict(torch.load('{}/model_{}'.format(CFG['model_path'], CFG['tag'])))
    test_pred_df = inference(model, device, test_df, CFG['test_img_path'])       
    test_pred_df.to_csv('kh_submission_raw.csv')

    # transform into submission format
    ids = []
    labels = []

    gp_mean = test_pred_df.loc[:, ['StudyInstanceUID']+CFG['exam_target_cols']].groupby('StudyInstanceUID', sort=False).mean()
    for col in CFG['exam_target_cols']:
        ids += [[patient+'_'+col for patient in gp_mean.index]]
        labels += [gp_mean[col].values]

    ids += [test_pred_df.SOPInstanceUID.values]
    labels += [test_pred_df[CFG['image_target_cols']].values[:,0]]
    ids = np.concatenate(ids)
    labels = np.concatenate(labels)

    assert len(ids) == len(labels)

    submission = pd.DataFrame()
    submission['id'] = ids
    submission['label'] = labels
    print(submission.head(10))
    submission.to_csv('submission.csv', index=False)