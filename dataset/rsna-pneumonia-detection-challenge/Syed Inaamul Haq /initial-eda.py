# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob 
import pylab 
import pydicom 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.de
df = pd.read_csv('../input/stage_1_train_labels.csv')
print (df.iloc[0])
patientId = df['patientId'][0]
dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
dcm_data = pydicom.dcmread(dcm_file)
print (dcm_data)
im = dcm_data.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)
pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('off')
def parse_data(df):
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]
    parsed = {}
    for n, row in df.iterrows(): 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom' : '../input/stage_1_train_images/%s.dcm' % pid,
                'label' : row['Target'],
                'boxes' : [] }
        if parsed[pid]['label']==1:
            parsed[pid]['boxes'].append(extract_box(row))
    return parsed
    
parsed = parse_data(df)
print(parsed['00436515-870c-4b36-a041-de91049b9ab4'])

def draw(data):
    d= pydicom.dcmread(data['dicom'])
    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')
    
def overlay_box(im, box, rgb, stroke=1):
 box = [int(b) for b in box]
 y1, x1, height, width = box 
 y2= y1+height
 x2= x1+width 
 
 im[y1:y1 + stroke, x1:x2] = rgb 
 im[y2:y2 + stroke, x1:x2] = rgb 
 im[y1:y2, stroke + x1:x2] = rgb 
 im[y1:y2, stroke + x1:x2] = rgb 
 
 return im 
 
draw(parsed['00436515-870c-4b36-a041-de91049b9ab4'])
df_detailed = pd.read_csv('../input/stage_1_detailed_class_info.csv')
print(df_detailed.iloc[0])

patientId = df_detailed['patientId'][0]
draw(parsed[patientId])

 
