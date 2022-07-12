# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#based on the following kernel: https://www.kaggle.com/hyeonho/pca-nusvc-0-95985


import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn import linear_model

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

oof = np.zeros(len(train))
preds = np.zeros(len(test))

for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=1).fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    skf = StratifiedKFold(n_splits=25, random_state=42)
    for train_index, test_index in skf.split(train2, train2['target']):

        clf = LGBMClassifier(
            nthread = 4,
            n_estimators = 2000,
            learning_rate = 0.01,
            num_leaves = 12,
            colsample_bytree = 1,
            subsample = 0.8,
            max_depth = -1,
            reg_alpha = 0,
            reg_lambda = 5,
            min_split_gain = 0.5,
            min_child_weight = 10,
            silent = 0,
            verbose = -1)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

auc = roc_auc_score(train['target'], oof)
print(f'AUC: {auc:.5}')

sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('logistic.csv', index=False)


