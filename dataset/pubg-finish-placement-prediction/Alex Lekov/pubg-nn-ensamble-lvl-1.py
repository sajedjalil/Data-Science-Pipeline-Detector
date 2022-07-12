import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir("../input/"))

df = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")

df_sub_other = pd.read_csv("../input/pubg-nn-part-other/submission_v2.csv")
df_sub_duo = pd.read_csv("../input/pubg-nn-part-duo-fpp/submission_v2.csv")
df_sub_squad = pd.read_csv("../input/pubg-nn-part-squad-fpp/submission_v2.csv")

print(df.shape), print(df_sub_other.shape)

fin_sub = np.zeros(len(df))
for i in range(len(df)):
    if 'squad-fpp' in str(df['matchType'][i]):
        fin_sub[i] = df_sub_squad['winPlacePerc'][i]
    elif 'duo-fpp' in str(df['matchType'][i]):
        fin_sub[i] = df_sub_duo['winPlacePerc'][i]
    else:
        fin_sub[i] = df_sub_other['winPlacePerc'][i]
        
sub_1 = df_sub_other.copy()
sub_1['winPlacePerc']=fin_sub

print(sub_1.shape), print(df_sub_other.shape)

print('part_2')

df_sub_other = pd.read_csv("../input/mod-simple-nn-baseline-4-py-v6/submission_v37.csv")

print(df_sub_other.shape), print(df_sub_other.shape)

print('blend')

fin_sub = sub_1.copy()
fin_sub['winPlacePerc']=(sub_1['winPlacePerc']+df_sub_other['winPlacePerc'])/2

print('save submission')

fin_sub.to_csv("submission_nn_ensamble_v5.csv", index=False)