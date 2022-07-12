import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir("../input/"))

df = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
df['matchType_mod'] = df['matchType'].apply(mapper)

df_sub_other = pd.read_csv("../input/pubg-lgb-part-other/submission_other_v3.csv")
df_sub_duo = pd.read_csv("../input/pubg-lgb-part-duo/submission_duo_v3.csv")
df_sub_squad = pd.read_csv("../input/pubg-lgb-part-squad/submission_squad_v3.csv")

print(df.shape), print(df_sub_other.shape)

fin_sub = np.zeros(len(df))
for i in range(len(df)):
    if 'squad' in str(df['matchType_mod'][i]):
        fin_sub[i] = df_sub_squad['winPlacePerc'][i]
    elif 'duo' in str(df['matchType_mod'][i]):
        fin_sub[i] = df_sub_duo['winPlacePerc'][i]
    else:
        fin_sub[i] = df_sub_other['winPlacePerc'][i]
        
sub = df_sub_other.copy()
sub['winPlacePerc']=fin_sub

print(sub.shape), print(df_sub_other.shape)

sub.to_csv("submission_v8.csv", index=False)