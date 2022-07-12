import subprocess
subprocess.run(["tar", "-xvf", "../input/gdcminstall/gdcm.tar"])
subprocess.run(["conda", "install", "../working/gdcm/conda-4.8.4-py37hc8dfbb8_2.tar.bz2"])
subprocess.run(["conda", "install", "../working/gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2"])
subprocess.run(["conda", "install", "../working/gdcm/libjpeg-turbo-2.0.3-h516909a_1.tar.bz2"])

import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import glob
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import sys
sys.path.insert(0, '../input/efficientnetpytorch/')
from efficientnet_pytorch import EfficientNet
sys.path.insert(0, '../input/pepytorchpretrained/')
from pretrainedmodels.senet import se_resnext101_32x4d, se_resnext50_32x4d
import pydicom

def correct_predictions(pred_prob_list, series_len_list):
    eps = 0.000001
    pred_prob_list_corrected = np.zeros(pred_prob_list.shape, dtype=np.float32)
    start = 0
    for i in tqdm(range(len(series_len_list))):
        end = series_len_list[i]
        negative_exam_for_pe = pred_prob_list[start+0]
        indeterminate = pred_prob_list[start+1]
        chronic_pe = pred_prob_list[start+2]
        acute_and_chronic_pe = pred_prob_list[start+3]
        central_pe = pred_prob_list[start+4]
        leftsided_pe = pred_prob_list[start+5]
        rightsided_pe = pred_prob_list[start+6]
        rv_lv_ratio_gte_1 = pred_prob_list[start+7]
        rv_lv_ratio_lt_1 = pred_prob_list[start+8]
        image_pe = pred_prob_list[start+9:end]

        loss_weight_list = np.zeros(pred_prob_list[start:end].shape, dtype=np.float32)
        loss_weight_list[0] = 0.0736196319
        loss_weight_list[1] = 0.09202453988
        loss_weight_list[2] = 0.1042944785
        loss_weight_list[3] = 0.1042944785
        loss_weight_list[4] = 0.1877300613
        loss_weight_list[5] = 0.06257668712
        loss_weight_list[6] = 0.06257668712
        loss_weight_list[7] = 0.2346625767
        loss_weight_list[8] = 0.0782208589
        loss_weight_list[9:] = 0.07361963*0.005
        
        if (np.amax(image_pe)<=0.5) and (int(negative_exam_for_pe>0.5)+int(indeterminate>0.5)==1) and (int(chronic_pe>0.5)+int(acute_and_chronic_pe>0.5)==0) and (int(central_pe>0.5)+int(leftsided_pe>0.5)+int(rightsided_pe>0.5)==0) and (int(rv_lv_ratio_gte_1>0.5)+int(rv_lv_ratio_lt_1>0.5)==0):
            pred_prob_list_corrected[start:end] = pred_prob_list[start:end]
        elif (np.amax(image_pe)>0.5) and (int(negative_exam_for_pe>0.5)+int(indeterminate>0.5)==0) and (int(chronic_pe>0.5)+int(acute_and_chronic_pe>0.5)<2) and (int(central_pe>0.5)+int(leftsided_pe>0.5)+int(rightsided_pe>0.5)>0) and (int(rv_lv_ratio_gte_1>0.5)+int(rv_lv_ratio_lt_1>0.5)==1):
            pred_prob_list_corrected[start:end] = pred_prob_list[start:end]
        else:
            to_neg = pred_prob_list[start:end].copy()
            for n in range(len(image_pe)):
                if image_pe[n]>0.5:
                    to_neg[9+n] = 0.5
            if negative_exam_for_pe>0.5 and indeterminate>0.5:
                if negative_exam_for_pe>indeterminate:
                    to_neg[1] = 0.5
                else:
                    to_neg[0] = 0.5
            elif negative_exam_for_pe<=0.5 and indeterminate<=0.5:
                if negative_exam_for_pe>indeterminate:
                    to_neg[0] = 0.5+eps
                else:
                    to_neg[1] = 0.5+eps
            if chronic_pe>0.5:
                to_neg[2] = 0.5
            if acute_and_chronic_pe>0.5:
                to_neg[3] = 0.5
            if central_pe>0.5:
                to_neg[4] = 0.5
            if leftsided_pe>0.5:
                to_neg[5] = 0.5
            if rightsided_pe>0.5:
                to_neg[6] = 0.5
            if rv_lv_ratio_gte_1>0.5:
                to_neg[7] = 0.5
            if rv_lv_ratio_lt_1>0.5:
                to_neg[8] = 0.5

            to_pos = pred_prob_list[start:end].copy()
            if np.amax(image_pe)<=0.5:
                max_idx = np.argmax(image_pe)
                to_pos[9+max_idx] = 0.5+eps
            if negative_exam_for_pe>0.5:
                to_pos[0] = 0.5
            if indeterminate>0.5:
                to_pos[1] = 0.5
            if chronic_pe>0.5 and acute_and_chronic_pe>0.5:
                if chronic_pe>acute_and_chronic_pe:
                    to_pos[3] = 0.5
                else:
                    to_pos[2] = 0.5
            if central_pe<=0.5 and leftsided_pe<=0.5 and rightsided_pe<=0.5:
                if central_pe>leftsided_pe and central_pe>rightsided_pe:
                    to_pos[4] = 0.5+eps
                if leftsided_pe>central_pe and leftsided_pe>rightsided_pe:
                    to_pos[5] = 0.5+eps
                if rightsided_pe>central_pe and rightsided_pe>leftsided_pe:
                    to_pos[6] = 0.5+eps
            if rv_lv_ratio_gte_1>0.5 and rv_lv_ratio_lt_1>0.5:
                if rv_lv_ratio_gte_1>rv_lv_ratio_lt_1:
                    to_pos[8] = 0.5
                else:
                    to_pos[7] = 0.5
            elif rv_lv_ratio_gte_1<=0.5 and rv_lv_ratio_lt_1<=0.5:
                if rv_lv_ratio_gte_1>rv_lv_ratio_lt_1:
                    to_pos[7] = 0.5+eps
                else:
                    to_pos[8] = 0.5+eps

            loss_weight_list1 = torch.tensor(loss_weight_list, dtype=torch.float32)
            pred_prob_list1 = torch.tensor(pred_prob_list[start:end], dtype=torch.float32)
            pred_prob_list_neg = torch.tensor(to_neg, dtype=torch.float32)
            pred_prob_list_pos = torch.tensor(to_pos, dtype=torch.float32)
            #print(loss_weight_list1.shape, pred_prob_list1.shape, pred_prob_list_neg.shape, pred_prob_list_pos.shape)
            to_neg_loss = ((torch.nn.BCELoss(reduction='none')(pred_prob_list1, pred_prob_list_neg)*loss_weight_list1).sum() / loss_weight_list1.sum()).numpy()
            to_pos_loss = ((torch.nn.BCELoss(reduction='none')(pred_prob_list1, pred_prob_list_pos)*loss_weight_list1).sum() / loss_weight_list1.sum()).numpy()

            if to_neg_loss>to_pos_loss:
                pred_prob_list_corrected[start:end] = to_pos
            else:
                pred_prob_list_corrected[start:end] = to_neg

        start = series_len_list[i]
    return pred_prob_list_corrected

def check_consistency(sub, test):
    
    '''
    Checks label consistency and returns the errors
    
    Args:
    sub   = submission dataframe (pandas)
    test  = test.csv dataframe (pandas)
    '''
    
    # EXAM LEVEL
    for i in test['StudyInstanceUID'].unique():
        df_tmp = sub.loc[sub.id.str.contains(i, regex = False)].reset_index(drop = True)
        df_tmp['StudyInstanceUID'] = df_tmp['id'].str.split('_').str[0]
        df_tmp['label_type']       = df_tmp['id'].str.split('_').str[1:].apply(lambda x: '_'.join(x))
        del df_tmp['id']
        if i == test['StudyInstanceUID'].unique()[0]:
            df = df_tmp.copy()
        else:
            df = pd.concat([df, df_tmp], axis = 0)
    df_exam = df.pivot(index = 'StudyInstanceUID', columns = 'label_type', values = 'label')
    
    # IMAGE LEVEL
    df_image = sub.loc[sub.id.isin(test.SOPInstanceUID)].reset_index(drop = True)
    df_image = df_image.merge(test, how = 'left', left_on = 'id', right_on = 'SOPInstanceUID')
    df_image.rename(columns = {"label": "pe_present_on_image"}, inplace = True)
    del df_image['id']
    
    # MERGER
    df = df_exam.merge(df_image, how = 'left', on = 'StudyInstanceUID')
    ids    = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
    labels = [c for c in df.columns if c not in ids]
    df = df[ids + labels]
    
    # SPLIT NEGATIVE AND POSITIVE EXAMS
    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())
    df_pos = df.loc[df.positive_images_in_exam >  0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]
    
    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS
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
    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 
                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)
    rule1d['broken_rule'] = '1d'

    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS
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
    
    # OUTPUT
    print('Found', len(errors), 'inconsistent predictions')
    return errors

####################################
df = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/test.csv')
study_id_list = df['StudyInstanceUID'].values
series_id_list = df['SeriesInstanceUID'].values
image_id_list = df['SOPInstanceUID'].values
series_list = []
series_dict = {}
image_dict = {}
for i in range(len(series_id_list)):
    series_id = study_id_list[i]+'_'+series_id_list[i]
    image_id = image_id_list[i]
    series_dict[series_id] = {'sorted_image_list': []}
    series_list.append(series_id)
    image_dict[image_id] = {'series_id': series_id, 'image_minus1': '', 'image_plus1': ''}
series_list = sorted(list(set(series_list)))
print(len(series_list), len(series_dict), len(image_dict))
######################################


#######################################
class BboxDataset(Dataset):
    def __init__(self, series_list):
        self.series_list = series_list
    def __len__(self):
        return len(self.series_list)
    def __getitem__(self,index):
        return index

class BboxCollator(object):
    def __init__(self, series_list):
        self.series_list = series_list
    def _window(self, x, WL=50, WW=350):
        upper, lower = WL+WW//2, WL-WW//2
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        return x
    def _load_dicom_array(self, f):
        dicom_files = glob.glob(os.path.join(f, '*.dcm'))
        dicoms = [pydicom.dcmread(d) for d in dicom_files]
        M = np.float32(dicoms[0].RescaleSlope)
        B = np.float32(dicoms[0].RescaleIntercept)
        z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
        sorted_idx = np.argsort(z_pos)
        dicom_files = np.asarray(dicom_files)[sorted_idx]
        dicoms = np.asarray(dicoms)[sorted_idx]
        selected_idx = [int(0.2*len(dicom_files)), int(0.3*len(dicom_files)), int(0.4*len(dicom_files)), int(0.5*len(dicom_files))]
        selected_dicom_files = dicom_files[selected_idx]
        selected_dicoms = dicoms[selected_idx]
        dicoms = np.asarray([d.pixel_array.astype(np.float32) for d in selected_dicoms])
        dicoms = dicoms * M
        dicoms = dicoms + B
        dicoms = self._window(dicoms, WL=100, WW=700)
        return dicoms, dicom_files
    def __call__(self, batch_idx):
        study_id = self.series_list[batch_idx[0]].split('_')[0]
        series_id = self.series_list[batch_idx[0]].split('_')[1]
        series_dir = '../input/rsna-str-pulmonary-embolism-detection/test/' + study_id + '/'+ series_id
        dicoms, dicom_files = self._load_dicom_array(series_dir)
        sorted_image_list = []
        for i in range(len(dicom_files)):
            name = dicom_files[i][-16:-4]
            sorted_image_list.append(name)
        x = np.zeros((4, 3, dicoms.shape[1], dicoms.shape[2]), dtype=np.float32)
        for i in range(4):
            x[i,0] = dicoms[i]
            x[i,1] = dicoms[i]
            x[i,2] = dicoms[i]
        return torch.from_numpy(x), sorted_image_list, self.series_list[batch_idx[0]]

class bbox_efficientnet(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = EfficientNet.from_name('efficientnet-b0')
        in_features = self.net._fc.in_features
        self.last_linear = nn.Linear(in_features, 4)
    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.net._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    
model_bbox = bbox_efficientnet()
model_bbox.load_state_dict(torch.load('../input/lungdetector/splitall/epoch34_polyak'))
model_bbox = model_bbox.cuda()
model_bbox.eval()

bbox_dict = {}

datagen = BboxDataset(series_list=series_list)
collate_fn = BboxCollator(series_list=series_list)
generator = DataLoader(dataset=datagen, collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
total_steps = len(generator)
for i, (images, sorted_image_list, series_id) in tqdm(enumerate(generator), total=total_steps):
    with torch.no_grad():
        start = i*4
        end = start+4
        if i == len(generator)-1:
            end = len(generator.dataset)*4
        images = images.cuda()
        logits = model_bbox(images)
        bbox = np.squeeze(logits.cpu().data.numpy())
        xmin = np.round(min([bbox[0,0], bbox[1,0], bbox[2,0], bbox[3,0]])*512)
        ymin = np.round(min([bbox[0,1], bbox[1,1], bbox[2,1], bbox[3,1]])*512)
        xmax = np.round(max([bbox[0,2], bbox[1,2], bbox[2,2], bbox[3,2]])*512)
        ymax = np.round(max([bbox[0,3], bbox[1,3], bbox[2,3], bbox[3,3]])*512)
        bbox_dict[series_id] = [int(max(0, xmin)), int(max(0, ymin)), int(min(512, xmax)), int(min(512, ymax))]
        for j in range(len(sorted_image_list)):
            name = sorted_image_list[j]
            if j==0:
                image_dict[name]['image_minus1'] = name
                image_dict[name]['image_plus1'] = sorted_image_list[j+1]
            elif j==len(sorted_image_list)-1:
                image_dict[name]['image_minus1'] = sorted_image_list[j-1]
                image_dict[name]['image_plus1'] = name
            else:
                image_dict[name]['image_minus1'] = sorted_image_list[j-1]
                image_dict[name]['image_plus1'] = sorted_image_list[j+1]
        series_dict[series_id]['sorted_image_list'] = sorted_image_list

print(len(bbox_dict), len(series_dict), len(image_dict))
print(bbox_dict[series_list[0]])
print(series_dict[series_list[0]])
print(image_dict[sorted_image_list[0]])
####################################


####################################
class PEDataset(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
    def _window(self, img, WL=50, WW=350):
        upper, lower = WL+WW//2, WL-WW//2
        X = np.clip(img.copy(), lower, upper)
        X = X - np.min(X)
        X = X / np.max(X)
        X = (X*255.0).astype('uint8')
        return X
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        data1 = pydicom.dcmread('../input/rsna-str-pulmonary-embolism-detection/test/'+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_minus1']+'.dcm')
        data2 = pydicom.dcmread('../input/rsna-str-pulmonary-embolism-detection/test/'+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        data3 = pydicom.dcmread('../input/rsna-str-pulmonary-embolism-detection/test/'+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_plus1']+'.dcm')
        x1 = data1.pixel_array
        x2 = data2.pixel_array
        x3 = data3.pixel_array
        x1 = x1*data1.RescaleSlope+data1.RescaleIntercept
        x2 = x2*data2.RescaleSlope+data2.RescaleIntercept
        x3 = x3*data3.RescaleSlope+data3.RescaleIntercept
        x1 = np.expand_dims(self._window(x1, WL=100, WW=700), axis=2)
        x2 = np.expand_dims(self._window(x2, WL=100, WW=700), axis=2)
        x3 = np.expand_dims(self._window(x3, WL=100, WW=700), axis=2)
        x = np.concatenate([x1, x2, x3], axis=2)
        bbox = self.bbox_dict[self.image_dict[self.image_list[index]]['series_id']]
        x = x[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        x = cv2.resize(x, (self.target_size,self.target_size))
        x = transforms.ToTensor()(x)
        x = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(x)
        return x
    
class pe_seresnext101(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        feature = x.view(x.size(0), -1)
        x = self.last_linear(feature)
        return feature, x
    
class pe_seresnext50(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = se_resnext50_32x4d(num_classes=1000, pretrained=None)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        feature = x.view(x.size(0), -1)
        x = self.last_linear(feature)
        return feature, x
    
class lv2Dataset(Dataset):
    def __init__(self,
                 feature_array,
                 feature_array1,
                 image_to_feature,
                 series_dict,
                 image_dict,
                 series_list,
                 seq_len):
        self.feature_array=feature_array
        self.feature_array1=feature_array1
        self.image_to_feature=image_to_feature
        self.series_dict=series_dict
        self.image_dict=image_dict
        self.series_list=series_list
        self.seq_len=seq_len
    def __len__(self):
        return len(self.series_list)
    def __getitem__(self,index):
        image_list = self.series_dict[self.series_list[index]]['sorted_image_list'] 
        if len(image_list)>self.seq_len:
            x = np.zeros((len(image_list), self.feature_array.shape[1]*3), dtype=np.float32)
            x1 = np.zeros((len(image_list), self.feature_array.shape[1]*3), dtype=np.float32)
            mask = np.ones((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]].copy() 
                x1[i,:self.feature_array.shape[1]] = self.feature_array1[self.image_to_feature[image_list[i]]].copy() 
            x = cv2.resize(x, (self.feature_array.shape[1]*3, self.seq_len), interpolation = cv2.INTER_LINEAR)
            x1 = cv2.resize(x1, (self.feature_array.shape[1]*3, self.seq_len), interpolation = cv2.INTER_LINEAR)
        else:
            x = np.zeros((self.seq_len, self.feature_array.shape[1]*3), dtype=np.float32)
            x1 = np.zeros((self.seq_len, self.feature_array.shape[1]*3), dtype=np.float32)
            mask = np.zeros((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]].copy()
                x1[i,:self.feature_array.shape[1]] = self.feature_array1[self.image_to_feature[image_list[i]]].copy()
                mask[i] = 1.
        x[1:,self.feature_array.shape[1]:self.feature_array.shape[1]*2] = x[1:,:self.feature_array.shape[1]] - x[:-1,:self.feature_array.shape[1]]
        x[:-1,self.feature_array.shape[1]*2:] = x[:-1,:self.feature_array.shape[1]] - x[1:,:self.feature_array.shape[1]]
        x = torch.tensor(x, dtype=torch.float32)
        x1[1:,self.feature_array.shape[1]:self.feature_array.shape[1]*2] = x1[1:,:self.feature_array.shape[1]] - x1[:-1,:self.feature_array.shape[1]]
        x1[:-1,self.feature_array.shape[1]*2:] = x1[:-1,:self.feature_array.shape[1]] - x1[1:,:self.feature_array.shape[1]]
        x1 = torch.tensor(x1, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return x, x1, mask, self.series_list[index]

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
# https://www.kaggle.com/bminixhofer/a-validation-framework-impact-of-the-random-seed
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class lv2Net(nn.Module):
    def __init__(self, input_len, lstm_size):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear_pe = nn.Linear(lstm_size*2, 1)
        self.last_linear_npe = nn.Linear(lstm_size*4, 1)
        self.last_linear_idt = nn.Linear(lstm_size*4, 1)
        self.last_linear_lpe = nn.Linear(lstm_size*4, 1)
        self.last_linear_rpe = nn.Linear(lstm_size*4, 1)
        self.last_linear_cpe = nn.Linear(lstm_size*4, 1)
        self.last_linear_gte = nn.Linear(lstm_size*4, 1)
        self.last_linear_lt = nn.Linear(lstm_size*4, 1)
        self.last_linear_chronic = nn.Linear(lstm_size*4, 1)
        self.last_linear_acute_and_chronic = nn.Linear(lstm_size*4, 1)
        self.attention = Attention(lstm_size*2, seq_len)
    def forward(self, x, mask):
        #x = SpatialDropout(0.5)(x)
        h_lstm1, _ = self.lstm1(x)
        logits_pe = self.last_linear_pe(h_lstm1)
        max_pool, _ = torch.max(h_lstm1, 1)
        att_pool = self.attention(h_lstm1, mask)
        conc = torch.cat((max_pool, att_pool), 1)
        logits_npe = self.last_linear_npe(conc)
        logits_idt = self.last_linear_idt(conc)
        logits_lpe = self.last_linear_lpe(conc)
        logits_rpe = self.last_linear_rpe(conc)
        logits_cpe = self.last_linear_cpe(conc)
        logits_gte = self.last_linear_gte(conc)
        logits_lt = self.last_linear_lt(conc)
        logits_chronic = self.last_linear_chronic(conc)
        logits_acute_and_chronic = self.last_linear_acute_and_chronic(conc)
        return logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic

seq_len = 384
feature_size = 2048*3
lstm_size = 512
image_size = 576
batch_size = 12
batch_size_lv2 = 8
num_partition = 16
    
model_pe = pe_seresnext50()
model_pe.load_state_dict(torch.load('../input/pedetector/seresnext50_splitall/epoch0'))
model_pe = model_pe.cuda()
model_pe.eval()

model_pe1 = pe_seresnext50()
model_pe1.load_state_dict(torch.load('../input/pedetector/seresnext50_split2/epoch0'))
model_pe1 = model_pe1.cuda()
model_pe1.eval()

model_lv2 = lv2Net(input_len=feature_size, lstm_size=lstm_size)
model_lv2.load_state_dict(torch.load('../input/lv2detector/splitall/seresnext50_384'))
model_lv2 = model_lv2.cuda()
model_lv2.eval()

model_lv21 = lv2Net(input_len=feature_size, lstm_size=lstm_size)
model_lv21.load_state_dict(torch.load('../input/lv2detector/split2/seresnext50_384'))
model_lv21 = model_lv21.cuda()
model_lv21.eval()

pred_prob_list = []
id_list = []
series_len_list = []
for part in range(num_partition):
    if part==num_partition-1:
        series_list_part = series_list[part*(len(series_list)//num_partition):]
    else:
        series_list_part = series_list[part*(len(series_list)//num_partition):(part+1)*(len(series_list)//num_partition)]
    
    image_list = []
    for series_id in series_list_part:
        image_list += list(series_dict[series_id]['sorted_image_list'])
    print(len(image_list), len(image_dict), len(bbox_dict))

    feature = np.zeros((len(image_list), 2048), dtype=np.float32)
    feature1 = np.zeros((len(image_list), 2048), dtype=np.float32)

    datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict, image_list=image_list, target_size=image_size)
    generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for i, images in tqdm(enumerate(generator), total=len(generator)):
        with torch.no_grad():
            start = i*batch_size
            end = start+batch_size
            if i == len(generator)-1:
                end = len(generator.dataset)
            images = images.cuda()
            features, logits = model_pe(images)
            features1, logits1 = model_pe1(images)
            feature[start:end] = np.squeeze(features.cpu().data.numpy())
            feature1[start:end] = np.squeeze(features1.cpu().data.numpy())
    print(feature.shape)

    image_to_feature = {}
    for i in range(len(feature)):
        image_to_feature[image_list[i]] = i
    
    datagen = lv2Dataset(feature_array=feature,
                         feature_array1=feature1,
                         image_to_feature=image_to_feature,
                         series_dict=series_dict,
                         image_dict=image_dict,
                         series_list=series_list_part,
                         seq_len=seq_len)
    generator = DataLoader(dataset=datagen,
                           batch_size=batch_size_lv2,
                           shuffle=False,
                           num_workers=2,
                           pin_memory=True)

    for j, (x, x1, mask, batch_series_list) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size_lv2
            end = start+batch_size_lv2
            if j == len(generator)-1:
                end = len(generator.dataset)
            x = x.cuda()
            x1 = x1.cuda()
            mask = mask.cuda()
            logits_pe, logits_npe, logits_idt, logits_lpe, logits_rpe, logits_cpe, logits_gte, logits_lt, logits_chronic, logits_acute_and_chronic = model_lv2(x, mask)
            logits_pe1, logits_npe1, logits_idt1, logits_lpe1, logits_rpe1, logits_cpe1, logits_gte1, logits_lt1, logits_chronic1, logits_acute_and_chronic1 = model_lv21(x1, mask)
            pred_prob_pe = 0.6*np.squeeze(logits_pe.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_pe1.sigmoid().cpu().data.numpy())
            pred_prob_npe = 0.6*np.squeeze(logits_npe.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_npe1.sigmoid().cpu().data.numpy())
            pred_prob_idt = 0.6*np.squeeze(logits_idt.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_idt1.sigmoid().cpu().data.numpy())
            pred_prob_lpe = 0.6*np.squeeze(logits_lpe.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_lpe1.sigmoid().cpu().data.numpy())
            pred_prob_rpe = 0.6*np.squeeze(logits_rpe.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_rpe1.sigmoid().cpu().data.numpy())
            pred_prob_cpe = 0.6*np.squeeze(logits_cpe.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_cpe1.sigmoid().cpu().data.numpy())
            pred_prob_chronic = 0.6*np.squeeze(logits_chronic.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_chronic1.sigmoid().cpu().data.numpy())
            pred_prob_acute_and_chronic = 0.6*np.squeeze(logits_acute_and_chronic.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_acute_and_chronic1.sigmoid().cpu().data.numpy())
            pred_prob_gte = 0.6*np.squeeze(logits_gte.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_gte1.sigmoid().cpu().data.numpy())
            pred_prob_lt = 0.6*np.squeeze(logits_lt.sigmoid().cpu().data.numpy()) + 0.4*np.squeeze(logits_lt1.sigmoid().cpu().data.numpy())
            for n in range(len(batch_series_list)):
                pred_prob_list.append(pred_prob_npe[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_negative_exam_for_pe')
                pred_prob_list.append(pred_prob_idt[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_indeterminate')
                pred_prob_list.append(pred_prob_chronic[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_chronic_pe')
                pred_prob_list.append(pred_prob_acute_and_chronic[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_acute_and_chronic_pe')
                pred_prob_list.append(pred_prob_cpe[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_central_pe')
                pred_prob_list.append(pred_prob_lpe[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_leftsided_pe')
                pred_prob_list.append(pred_prob_rpe[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_rightsided_pe')
                pred_prob_list.append(pred_prob_gte[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_rv_lv_ratio_gte_1')
                pred_prob_list.append(pred_prob_lt[n])
                id_list.append(batch_series_list[n].split('_')[0]+'_rv_lv_ratio_lt_1')
                num_image = len(series_dict[batch_series_list[n]]['sorted_image_list'])
                if num_image>seq_len:
                    pred_prob_list += list(np.squeeze(cv2.resize(pred_prob_pe[n, :], (1, num_image), interpolation = cv2.INTER_LINEAR)))
                else:
                    pred_prob_list += list(pred_prob_pe[n, :num_image])
                id_list += list(series_dict[batch_series_list[n]]['sorted_image_list'])
                series_len_list.append(len(id_list))
                    
print(len(id_list), len(pred_prob_list), len(series_len_list))
print(id_list[:5])
print(pred_prob_list[:5])
print(series_len_list[:5])

pred_prob_list = np.array(pred_prob_list)
series_len_list = np.array(series_len_list)
pred_prob_list = correct_predictions(pred_prob_list, series_len_list)
print(len(id_list), len(pred_prob_list), len(series_len_list))


############################################

sub_df = pd.DataFrame(data={'id': id_list, 'label': pred_prob_list})
errors = check_consistency(sub_df, df)
if len(errors)==0:
    sub_df.to_csv('submission.csv', index=False)