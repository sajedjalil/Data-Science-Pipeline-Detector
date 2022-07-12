import numpy as np
import pandas as pd 
print("===================================Start Run===================================")
dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }

eco_train = pd.read_csv('../input/train.csv', dtype=dtypes)[['app','os','channel','is_attributed']]
#mean_aoc_train = eco_train.groupby(['app','os','channel'])['is_attributed'].mean().reset_index()
#mean_ac_train = eco_train.groupby(['app','channel'])['is_attributed'].mean().reset_index()
mean_attributed = eco_train['is_attributed'].mean()
eco_test = pd.read_csv('../input/test.csv', dtype=dtypes)[['click_id','app','os','channel']]

print("===================================Success input data===================================")

merge_aoc_trte = pd.merge(eco_test, eco_train.groupby(['app','os','channel'])['is_attributed'].mean().reset_index(), on=['app','os','channel'], how='left')
#merge_trte_null =  merge_aoc_trte[merge_aoc_trte.is_attributed.isnull()].drop('is_attributed', axis=1)
merge_ac_trte = pd.merge(merge_aoc_trte[merge_aoc_trte.is_attributed.isnull()].drop('is_attributed', axis=1), eco_train.groupby(['app','channel'])['is_attributed'].mean().reset_index(), on=['app','channel'], how='left').fillna(mean_attributed)
merge_trte = pd.concat([merge_aoc_trte.dropna(subset=['is_attributed']), merge_ac_trte], ignore_index = True)
print("===================================Print -merge_trte-===================================")
print(merge_trte)

submit = merge_trte[['click_id', 'is_attributed']]
print("===================================Print -submit-===================================")
print(submit)
submit.to_csv('submit.csv', index=False)