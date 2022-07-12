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
df1 = pd.read_csv('../input/talkingdata-wordbatch-fm-ftrl-lb-0-9769/wordbatch_fm_ftrl.csv')
df2 = pd.read_csv('../input/talkingdata-added-new-features-in-lightgbm/sub_it7.csv')
df3 = pd.read_csv('../input/krishna-s-lgbm-to-catboost-undrsmpl-1-1/Krishna_s_CatBoost_1_1_CB_1_1.csv')
df4 = pd.read_csv('../input/log-and-harmonic-mean-lets-go/submission_avg.csv')
df5 = pd.read_csv('../input/log-and-harmonic-mean-lets-go/submission_geo.csv')
df6 = pd.read_csv('../input/if-you-run-on-entire-dataset-lb-0-9798/sub_it6.csv')


models = {
            'df1' : {
                    'name':'wordbatch_fm_ftrl',
                    'score':97.69,
                    'df':df1 },
            'df3' :{'name':'Krishna_s_CatBoost_1_1_CB_1_1',
                    'score':97.33,
                    'df':df3 },
            'df2' :{'name':'sub_it7',
                    'score':96.33,
                    'df':df2 },    
            'df4' :{'name':'submission_avg',
                    'score':97.87,
                    'df':df4 }, 
            #'df5' :{'name':'submission_geo',
            #        'score':1,
            #        'df':df5 }, 
            'df6' :{'name':'submission_log4',
                    'score':97.98,
                    'df':df6 },         
         }

df1.head()         

isa_lg = 0
isa_hm = 0
isa_am=0

print("Blending...\n")
for df in models.keys() : 
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
    isa_am +=models[df]['df'].is_attributed

NUM_MODELS = len(models.keys())
isa_lg = np.exp(isa_lg/NUM_MODELS)
isa_hm = NUM_MODELS/isa_hm
isa_am=isa_am/NUM_MODELS

print("Isa log\n")
print(isa_lg[:5])
print()
print("Isa harmo\n")
print(isa_hm[:5])


sub_hm = pd.DataFrame()
sub_hm['click_id'] = df1['click_id']
sub_hm['is_attributed'] = isa_hm
print(sub_hm.head())
print('hm min', isa_hm.min())
print('hm max', isa_hm.max())

isa_fin=(isa_am+isa_hm+isa_lg)/3
print('isa min', isa_fin.min())
print('isa max', isa_fin.max())

sub_fin = pd.DataFrame()
sub_fin['click_id'] = df1['click_id']
sub_fin['is_attributed'] = np.clip(isa_fin, 0., 1.)

print("Writing...")

#sub_hm.to_csv('submission_hm.csv', index=False, float_format='%.9f')
sub_fin.to_csv('submission.csv', index=False, float_format='%.9f')
