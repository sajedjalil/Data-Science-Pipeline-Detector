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
df1 = pd.read_csv('../input/sub-log12/submission_log12.csv')
df2 = pd.read_csv('../input/lightgbm-lets-go/submission_from_kernal15.csv')
df3 = pd.read_csv('../input/submission-9780/submission_9780.csv')
df4 = pd.read_csv('../input/log-hm/submission_hm.csv')
df5 = pd.read_csv('../input/log-hm/submission_log.csv')

models = {
          'df2' : {
                    'name':'log12',
                    'score':97.92,
                    'df':df1 },
    
           'df2' : {
                    'name':'lightgbm86',
                    'score':97.86,
                    'df':df2 },
           'df3' : {
                    'name':'lightgbm80',
                    'score':97.80,
                    'df':df3 }
            
        '''  'df4' : {
                    'name':'hm87',
                    'score':97.87,
                    'df':df4 },
            
           'df5' : {
                    'name':'log87',
                    'score':1,
                    'df':df5 }     '''   
         }

df2.head()         

isa_lg = 0
isa_hm = 0

print("Blending...\n")
for df in models.keys() : 
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
isa_lg = np.exp(isa_lg/3)
isa_hm = 1/isa_hm

print("Isa log\n")
print(isa_lg[:5])
print()
print("Isa harmo\n")
print(isa_hm[:5])

sub_log = pd.DataFrame()
sub_log['click_id'] = df2['click_id']
sub_log['is_attributed'] = (isa_hm+isa_lg)/2
sub_log.head()

sub_hm = pd.DataFrame()
sub_hm['click_id'] = df2['click_id']
sub_hm['is_attributed'] = isa_hm
sub_hm.head()

print("Writing...")
sub_log.to_csv('submission_log14.csv', index=False, float_format='%.9f')
#sub_hm.to_csv('submission_hm4.csv', index=False, float_format='%.9f')