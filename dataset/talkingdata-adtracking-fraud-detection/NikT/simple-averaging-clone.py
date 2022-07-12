# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print("Reading the data...\n")
df1 = pd.read_csv('../input/if-you-run-on-entire-dataset-lb-0-9798/sub_it6.csv')
df2 = pd.read_csv('../input/entire-dataset-lb-0-9811-lightgbm/sub-it200102csv')
df3 = pd.read_csv('../input/lgb-87mil-rows-for-training/lgb87mil.csv')
df4 = pd.read_csv('../input/log-and-harmonic-mean-lets-go/submission_avg.csv')
#df5 = pd.read_csv('../input/log-and-harmonic-mean-lets-go/submission_geo.csv')
#df6 = pd.read_csv('../input/notebook-version-of-talkingdata-lb-0-9786/sub_it24.csv')
#df7 = pd.read_csv('../input/simple-mix-lb-09780-sub-log/sub_log.csv')
#df8 = pd.read_csv('../input/simple-mix-lb-09780-sub-log/sub_hm.csv')
#df9 = pd.read_csv('../input/simple-linear-stacking-with-ranks-lb-0-9760/sub_stacked.csv')

models = {
    'df1': {
        'weight': 0.2553,
        'score': 97.98,
        'df': df1
    },
    'df2': {
        'weight': 0.5319,
        'score': 98.11,
        'df': df2
    },  
    'df3': {
        'weight': 0.1915,
        'score': 97.95,
        'df': df3
    },
    'df4': {
        'weight': 0.0213,
        'score': 97.87,
        'df': df4
    }, 
   
}

count_models = len(models)  

isa_lg = 0
isa_hm = 0
isa_am = 0
isa_gm=0
print("Blending...\n")
for df in models.keys() : 
    isa_lg += models[df]['weight'] * np.log(models[df]['df'].is_attributed)
    isa_hm += models[df]['weight']/(models[df]['df'].is_attributed)
    isa_am += isa_am
    isa_gm *= isa_gm
isa_lg = np.exp(isa_lg)
isa_hm = 1/isa_hm
isa_am = isa_am/count_models
isa_gm = (isa_gm)**(1/count_models)

print("Isa log\n")
print(isa_lg[:count_models])
print()
print("Isa harmo\n")
print(isa_hm[:count_models])

sub_log = pd.DataFrame()
sub_log['click_id'] = df1['click_id']
sub_log['is_attributed'] = isa_lg
sub_log.head()

sub_hm = pd.DataFrame()
sub_hm['click_id'] = df1['click_id']
sub_hm['is_attributed'] = isa_hm
sub_hm.head()

sub_fin=pd.DataFrame()
sub_fin['click_id']=df1['click_id']
sub_fin['is_attributed']= (7*isa_lg+3*isa_hm)/10

print("Writing...")
#sub_log.to_csv('submission_log2.csv', index=False, float_format='%.9f')
#sub_hm.to_csv('submission_hm2.csv', index=False, float_format='%.9f')
sub_fin.to_csv('submission_final_f.csv', index=False, float_format='%.9f')
