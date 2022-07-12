# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

"""
Speech Recognition competition.
To avoid leakage between train and validation sets the training data
needs to be split by user, so that all files for any user only
occur in either the train set or validation set, but not both.
This script is a template for splitting the training data by user.
The trndf and tardf dataframes which are created, contain the train
and validation filepaths for each fold.
"""

import os
import glob
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold

trainAudioPath = '../input/train/audio'

allFilepaths = glob.glob(trainAudioPath+'/*/*.wav')
p = re.compile(r'/([^/]+)/([^_/]+)_\w+\.wav$')
filepaths = []
labels = []
userIds = []

for fp in allFilepaths:
  m = p.search(fp)
  label,userId = m.groups()
  if label=='_background_noise_': continue
  filepaths.append(fp)
  labels.append(label)
  userIds.append(userId)
    
traindf = pd.DataFrame({'userId':userIds,'filepath':filepaths,'label':labels})

users = traindf.userId.drop_duplicates()

nFolds = 3


for foldNum,(trninds,tarinds) in enumerate(KFold(n_splits=nFolds,shuffle=True,random_state=1234).split(users)):
  trnUserIds = users.iloc[trninds]
  tarUserIds = users.iloc[tarinds]
  
  #trndf and tardf contain the train/validation instances for the current fold
  trndf = traindf[traindf.userId.isin(trnUserIds)]
  tardf = traindf[traindf.userId.isin(tarUserIds)]
  
  #code to train each fold of classifier can go here
  print('')
  print('train fold',foldNum)
  print(trndf.head())
  print('')  
  print('val fold',foldNum)
  print(tardf.head()) 
  