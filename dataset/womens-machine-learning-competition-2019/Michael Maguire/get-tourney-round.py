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
def game_round(sub):
    R0 = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    R1 = [(1,16,8,9),(5,12,4,13),(6,11,3,14),(7,10,2,15)]
    R2 = [(1,16,8,9,5,12,4,13),(6,11,3,14,7,10,2,15)]
    R4 = [('W','X'),('Y','Z')]
    R5 = [('W','Y'),('W','Z'),('X','Y'),('X','Z')]
    Rnds = np.zeros(sub.shape[0])
    for S in R0:
        Rnds += list(map(lambda w,x,y,z: 1 if (w==x)&(y in S)&(z in S) else 0, sub['ARegion'], sub['BRegion'], sub['ASeed'], sub['BSeed']))    
    sub['previous'] = Rnds
    for S in R1:
        Rnds += list(map(lambda p,w,x,y,z: 2 if (p==0)&(w==x)&(y in S)&(z in S) else 0, sub['previous'], sub['ARegion'], sub['BRegion'], sub['ASeed'], sub['BSeed']))
    sub['previous'] = Rnds
    for S in R2:
        Rnds += list(map(lambda p,w,x,y,z: 3 if (p==0)&(w==x)&(y in S)&(z in S) else 0, sub['previous'], sub['ARegion'], sub['BRegion'], sub['ASeed'], sub['BSeed']))
    Rnds += list(map(lambda w,x,y,z: 4 if (w==x)&( ((y in R2[0])&(z in R2[1]))|((y in R2[1])&(z in R2[0])) ) else 0, sub['ARegion'], sub['BRegion'], sub['ASeed'], sub['BSeed']))
    
    for S in R4:
        Rnds += list(map(lambda x,y: 5 if (x!=y)&(x in S)&(y in S) else 0, sub['ARegion'], sub['BRegion']))
    for S in R5:
        Rnds += list(map(lambda x,y: 6 if (x!=y)&(x in S)&(y in S) else 0, sub['ARegion'], sub['BRegion']))
    sub['Round'] = Rnds
    sub.drop('previous', axis=1, inplace=True)
    sub['DayNum'] = [133]*sub.shape[0]
    return sub

submission = pd.read_csv('../input/WSampleSubmissionStage2.csv')
submission['Season']  = submission['ID'].apply(lambda x: int(x[0:4]))
submission['ATeamID'] = submission['ID'].apply(lambda x: int(x[5:9]))
submission['BTeamID'] = submission['ID'].apply(lambda x: int(x[10:]))
seed = pd.read_csv('../input/stage2wdatafiles/WNCAATourneySeeds.csv')
seed['ATeamID'] = seed['TeamID']
seed['BTeamID'] = seed['TeamID']
seed['ASeed'] = seed['Seed'].apply(lambda x: int(x[1:3]))
seed['BSeed'] = seed['ASeed']
seed['ARegion'] = seed['Seed'].apply(lambda x:x[0])
seed['BRegion'] = seed['ARegion']
submission = submission.merge(seed[['Season','ATeamID','ASeed','ARegion']], on=['Season','ATeamID'], how='left')
submission = submission.merge(seed[['Season','BTeamID','BSeed','BRegion']], on=['Season','BTeamID'], how='left')


submission = game_round(submission)