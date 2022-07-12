# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats, optimize, interpolate
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
pd.set_option('display.max_colwidth', 1000)

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
holdout = pd.read_csv("../input/test.csv")
X = train.iloc[:,2:].values
y = train['target'].values

holdout_X = holdout.iloc[:,1:].values
holdout_ids = holdout["ID_code"]

###################################################  Scale data  ############################################

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
holdout_scale = min_max_scaler.fit_transform(holdout_X)

################################################### Run Model and cross validation ############################################
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm.fit(X_scale, y)

################################################### Submission file  ############################################
holdout_predictions = lgbm.predict_proba(holdout_scale).tolist()
pred = []
for i in range(len(holdout_predictions)):
    pred.append(holdout_predictions[i][0])
    
submission = pd.DataFrame({"ID_code": holdout_ids,"target": pred})
submission.to_csv('sample_submission.csv', index = False)




for i in sample.iloc[:, 3:10]:
    sns.lmplot(x=sample.iloc[:, 3:10], y="var_2", hue="target", data=sample)
    
    