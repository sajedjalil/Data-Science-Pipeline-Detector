# coding: utf-8


'''

Using locations reader from: Jonne - CNN Segmentation + connected components
    https://www.kaggle.com/jonnedtc/cnn-segmentation-connected-components
    
Using features from: Q&A with Only Pictuers! (RSNA Pneumonia Detection Challenge)
    version 35: https://www.kaggle.com/thomasjpfan/q-a-with-only-pictures?scriptVersionId=5454972

I added a few extra categorical features, like binning into a xy_signature:
    
    100
    100    ---> this 9 digits signature means that the patient has a RoI on the "upper-left" and other one in the "middle-left"
    000
    
Or aspect_ratio_signature:
    
    0000010001 --> the x-ray has a bounding box more width than height (last decile); and other one more close to the median aspect_ratio

Also binned the RoI area average per patient

The output are two CSVs: train_features.csv and 5_folds_rsna-pneumonia-detection-challenge.csv

'''
NUM_FOLDS=5
NUM_BINS=10
NUM_XY_BINS=3
RANDOM_SEED=1234



import sklearn
print("sklearn version: ", sklearn.__version__)

features={}


import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize


#from matplotlib import pyplot as plt
#plt.ioff() #disable interactiveness in matplotlib.pyplot commands

from scipy import stats
from sklearn.model_selection import StratifiedKFold

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


bins=list(np.linspace(0, 100, num=NUM_BINS+1))
xy_bins=list(np.linspace(0, 100, num=NUM_XY_BINS+1))


# empty dictionary
pneumonia_locations = {}
patient_ids=set()
aspect_ratios=[]
x_positions=[]
y_positions=[]
# load table
with open(os.path.join('../input/stage_1_train_labels.csv'), mode='r') as infile:
    # open reader
    reader = csv.reader(infile)
    # skip header
    next(reader, None)
    # loop through rows
    for rows in reader:
        # retrieve information
        
        filename = rows[0]
        patient_ids.add(filename)
        location = rows[1:5]
        pneumonia = rows[5]
        # if row contains pneumonia add label to dictionary
        # which contains a list of pneumonia locations per filename
        if pneumonia == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            x,y,width,height=location
            aspect_ratios.append(width/height)
            x_positions.append(x)
            y_positions.append(y)
            # save pneumonia location in dictionary
            if filename in pneumonia_locations:
                pneumonia_locations[filename].append(location)
            else:
                pneumonia_locations[filename] = [location]


patient_ids=list(patient_ids)


class_info_df = pd.read_csv('../input/stage_1_detailed_class_info.csv')

for patient_id in sorted(set(patient_ids)):
    has_pneumonia=int(patient_id in pneumonia_locations)
    rois=[] if not has_pneumonia else pneumonia_locations[patient_id]
    features[patient_id]={
        'patientId':patient_id,
#        'pneumonia': has_pneumonia,
        'num_rois': len(rois),
    }
    
    area=0
    aspect_ratio_signature=["0"]*NUM_BINS
    xy_signature=["0"]*NUM_XY_BINS*NUM_XY_BINS
    for x,y,width,height in rois:
        area+=width*height
        
        aspect_ratio=width/height
        aspect_ratio_percentile = stats.percentileofscore(aspect_ratios, aspect_ratio)
        aspect_ratio_signature[int(np.digitize(aspect_ratio_percentile,bins,right=True))-1]='1'
        
        x_percentile = stats.percentileofscore(x_positions, x)
        y_percentile = stats.percentileofscore(x_positions, y)
        xy_signature[int(np.digitize(x_percentile,xy_bins,right=True))-1 + \
                     int(np.digitize(y_percentile,xy_bins,right=True))-1 * NUM_XY_BINS ] = '1'
        
    classes=sorted(set(class_info_df[class_info_df.patientId==patient_id]['class'].values))
    classes_str=' AND '.join(classes)
    features[patient_id]['rois_area']=area
    features[patient_id]['rois_area_avg']=0 if len(rois)==0 else area/len(rois)
    features[patient_id]['aspect_ratio_signature']=''.join(aspect_ratio_signature)
    features[patient_id]['xy_signature']=''.join(xy_signature)
    features[patient_id]['classes']=classes_str
#    if has_pneumonia:
#        1/0




features_df=pd.DataFrame(list(features.values()))
#features_df=features_df.merge(class_info_df, how='left', on='patientId')



'''

FEATURES FROM KERNEL: Q&A with Only Pictuers! (RSNA Pneumonia Detection Challenge)

'''
print('FEATURES FROM KERNEL: Q&A with Only Pictuers! (RSNA Pneumonia Detection Challenge')

import seaborn as sns
import pandas as pd
import pydicom
import numpy as np
#import warnings
import multiprocessing
import os



# Get all data
tr = pd.read_csv('../input/stage_1_train_labels.csv')
tr['aspect_ratio'] = (tr['width']/tr['height'])
tr['area'] = tr['width'] * tr['height']

def get_info(patientId, root_dir='../input/stage_1_train_images/'):
    fn = os.path.join(root_dir, f'{patientId}.dcm')
    dcm_data = pydicom.read_file(fn)
    return {'age': dcm_data.PatientAge, 
            'gender': dcm_data.PatientSex, 
            'id': os.path.basename(fn).split('.')[0],
            'pixel_spacing': float(dcm_data.PixelSpacing[0]),
            'mean_black_pixels': np.mean(dcm_data.pixel_array == 0)}

patient_ids = list(tr.patientId.unique())
with multiprocessing.Pool(4) as pool:
    result = pool.map(get_info, patient_ids)
    
demo = pd.DataFrame(result)
demo['gender'] = demo['gender'].astype('category')
demo['age'] = demo['age'].astype(int)

tr = (tr.merge(demo, left_on='patientId', right_on='id', how='left')
        .drop(columns='id'))




centers = (tr.dropna(subset=['x'])
           .assign(center_x=tr.x + tr.width / 2, center_y=tr.y + tr.height / 2))


areas = tr.dropna(subset=['area'])


from sklearn.mixture import GaussianMixture
clf = GaussianMixture(n_components=2, random_state=RANDOM_SEED)
clf.fit(centers[['center_x', 'center_y']])
center_probs = clf.predict_proba(centers[['center_x', 'center_y']])
Z = -clf.score_samples(centers[['center_x', 'center_y']])
outliers = set(centers.iloc[Z > 17]['patientId'].values)

high_black_pixel_patientIds = tr.loc[tr.mean_black_pixels > 0.55, 'patientId'].drop_duplicates()
high_white_pixel_patientIds = tr.loc[tr.mean_black_pixels < 0.000001, 'patientId'].drop_duplicates()

tr['has_outliers']=tr['patientId'].isin(outliers)
tr['high_black_pixel']=tr['patientId'].isin(high_black_pixel_patientIds)
tr['high_white_pixel_patientIds']=tr['patientId'].isin(high_white_pixel_patientIds)

tr.drop(['x', 'y', 'width', 'height', 'Target', 'aspect_ratio', 'area'], axis=1, inplace=True)

features_df.drop(['rois_area'], axis=1, inplace=True)

tr=tr.drop_duplicates(subset=['patientId'])
features_df=features_df.merge(tr, how='left', on='patientId')
print(features_df.shape)
print(tr.shape)


num_observations=len(os.listdir('../input/stage_1_train_images'))
assert num_observations==features_df.shape[0]
features_df.to_csv('train_features.csv', index=False, header=True)

def to_binned_feature(features_df, col_name):
    values_gt_0=features_df[features_df[col_name]>0.0][col_name].values
    scores = [stats.percentileofscore(values_gt_0, x, kind='rank') for x in features_df[col_name].values]
    binned_feature=np.digitize(scores,bins,right=True)-1
    return binned_feature

features_df['rois_area_avg']= to_binned_feature(features_df, 'rois_area_avg')
features_df['mean_black_pixels']= to_binned_feature(features_df, 'mean_black_pixels')
features_df.loc[features_df['age']>100, 'age']=-1
features_df['age']= to_binned_feature(features_df, 'age')

#skf = StratifiedKFold(n_splits=NUM_FOLDS, random_state=RANDOM_SEED)



features_df['classes2']=features_df['classes']+features_df['num_rois'].astype(str)


features_df.sort_values(inplace=True, by=[
    'classes2', # class + num_rois
    'aspect_ratio_signature',
    'xy_signature',
    'rois_area_avg',
    'gender',
    'age',
    'mean_black_pixels',
    'pixel_spacing',
    'has_outliers',
    'high_black_pixel',
    'high_white_pixel_patientIds'
])

features_df['fold']=(list(range(NUM_FOLDS))*(num_observations//NUM_FOLDS+1))[:num_observations]


for col_name in ['classes', 'num_rois',  'aspect_ratio_signature', 'xy_signature', 'rois_area_avg', 'gender', 'age', 'mean_black_pixels', 'pixel_spacing', 'has_outliers', 'high_black_pixel', 'high_white_pixel_patientIds']:
    print("\nCount by {}:\n{}".format(col_name, "="*35))
    print (features_df.groupby([col_name, 'fold'])['patientId'].count())

features_df[['patientId', 'fold']].to_csv('{}_folds_rsna-pneumonia-detection-challenge.csv'.format(NUM_FOLDS), index=False, header=True)
