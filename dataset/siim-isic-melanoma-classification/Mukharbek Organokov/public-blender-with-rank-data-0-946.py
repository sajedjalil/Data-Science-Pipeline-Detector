# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        continue

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

## PLEASE UPVOTE ORIGINAL KERNELS!
## CREDIT GOES TO:

## KERNELS:  LINKS:                                                                                                  
## 866:      https://www.kaggle.com/ajaykumar7778/inceptionresnetv2-tpu                      (use v7)                                             
## 877:      https://www.kaggle.com/yeayates21/siim-keras-efficientnetb3-starter-tfrec-tpu   (use v7)                          
## 879:      https://www.kaggle.com/arroqc/siim-isic-pytorch-lightning-starter-seresnext50   (use v2)
## 884:      https://www.kaggle.com/yasufuminakama/tpu-siim-isic-efficientnetb3-inference    (use v2)                           
## 892:      https://www.kaggle.com/khoongweihao/siim-isic-multiple-model-training-inference (use v4)
## 897:      https://www.kaggle.com/soham1024/melanoma-efficientnetb6-inference              (use v14)                                      
## 910:      https://www.kaggle.com/ajaykumar7778/melanoma-tpu-efficientnet-b5-dense-head    (use v1)
## 914:      https://www.kaggle.com/cdeotte/image-and-tabular-data-0-915                     (use v1)                                         
## 927:      https://www.kaggle.com/shonenkov/inference-single-model-melanoma-starter        (use v2)                        

from pathlib import Path

sub_path = Path("../input/melanoma-public")
sub_866_path = sub_path/'submission_866.csv'
sub_877_path = sub_path/'submission_877.csv'
sub_879_path = sub_path/'submission_879.csv'
sub_884_path = sub_path/'submission_884.csv'
sub_892_path = sub_path/'submission_892.csv'
sub_897_path = sub_path/'submission_897.csv'
sub_910_path = sub_path/'submission_910.csv'
sub_914_path = sub_path/'submission_914.csv'
sub_927_path = sub_path/'submission_927.csv'

print(sub_910_path)
#sub_910.head(3)

sub_866 = pd.read_csv(sub_866_path)
sub_877 = pd.read_csv(sub_877_path)
sub_879 = pd.read_csv(sub_879_path)
sub_884 = pd.read_csv(sub_884_path)
sub_892 = pd.read_csv(sub_892_path)
sub_897 = pd.read_csv(sub_897_path)
sub_910 = pd.read_csv(sub_910_path)
sub_914 = pd.read_csv(sub_914_path)
sub_927 = pd.read_csv(sub_927_path)


sub_866 = sub_866.sort_values(by="image_name")
sub_877 = sub_877.sort_values(by="image_name")
sub_879 = sub_879.sort_values(by="image_name")
sub_884 = sub_884.sort_values(by="image_name")
sub_892 = sub_892.sort_values(by="image_name")
sub_897 = sub_897.sort_values(by="image_name")
sub_910 = sub_910.sort_values(by="image_name")
sub_914 = sub_914.sort_values(by="image_name")
sub_927 = sub_927.sort_values(by="image_name")

## No Rank 
"""
out1 = sub_866["target"].astype(float).values
out2 = sub_877["target"].astype(float).values
out3 = sub_879["target"].astype(float).values
out4 = sub_884["target"].astype(float).values
out5 = sub_892["target"].astype(float).values
out6 = sub_897["target"].astype(float).values
out7 = sub_910["target"].astype(float).values
out8 = sub_914["target"].astype(float).values
out9 = sub_927["target"].astype(float).values
"""

## Rank
from scipy.stats import rankdata
out1 = rankdata(sub_866["target"].astype(float).values)
out2 = rankdata(sub_877["target"].astype(float).values)
out3 = rankdata(sub_879["target"].astype(float).values)
out4 = rankdata(sub_884["target"].astype(float).values)
out5 = rankdata(sub_892["target"].astype(float).values)
out6 = rankdata(sub_897["target"].astype(float).values)
out7 = rankdata(sub_910["target"].astype(float).values)
out8 = rankdata(sub_914["target"].astype(float).values)
out9 = rankdata(sub_927["target"].astype(float).values)

merge_output = []

# Dummy weights, find your strategy!

n=9

w1 = 0.00
w2 = 0.00
w3 = 0.00
w4 = 0.00
w5 = 0.00
w6 = 0.10 
w7 = 0.15 
w8 = 0.30
w9 = 0.45

print('Sum weights:',w1+w2+w3+w4+w5+w6+w7+w8+w9)


for o1, o2, o3, o4, o5, o6, o7, o8, o9 in zip(out1, out2, out3, out4, out5, out6, out7, out8, out9):
    #print(o1,type(o1))
    o = float(o1*w1 + o2*w2 + o3*w3 + o4*w4 + o5*w5 + o6*w6 + o7*w7 + o8*w8 + o9*w9)
    merge_output.append(o)

    
sub_866["target"] = merge_output
sub_866["target"] = sub_866["target"].astype(float)
#sub_866 = sub_866.drop(['index'], axis=1)
sub_866.to_csv("submission.csv", index=False)

sub_866.head(3)