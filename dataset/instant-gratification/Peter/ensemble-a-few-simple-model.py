## __author__ Peter (https://www.kaggle.com/pestipeti)

## The important part is from Chris Deotte's kernel
## https://www.kaggle.com/cdeotte/support-vector-machine-0-925

import os
import numpy as np
import pandas as pd
import random as rn

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier

SEED = 5667

np.random.seed(SEED)
rn.seed(SEED)

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')

def execute_classifier(clf, n_folds=15):
    
    # INITIALIZE VARIABLES
    _oof = np.zeros(len(train_df))
    _preds = np.zeros(len(test_df))
    cols = [c for c in train_df.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
    
    # BUILD 512 SEPARATE NON-LINEAR MODELS
    for i in range(512):
    
        # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I
        train_df2 = train_df[train_df['wheezy-copper-turtle-magic'] == i]
        test_df2 = test_df[test_df['wheezy-copper-turtle-magic'] == i]
        idx1 = train_df2.index
        idx2 = test_df2.index
        train_df2.reset_index(drop=True, inplace=True)
    
        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
        sel = VarianceThreshold(threshold=1.5).fit(train_df2[cols])
        train_df3 = sel.transform(train_df2[cols])
        test_df3 = sel.transform(test_df2[cols])

        # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
        skf = StratifiedKFold(n_splits=n_folds, random_state=SEED)

        for train_df_index, test_df_index in skf.split(train_df3, train_df2['target']):
            clf.fit(train_df3[train_df_index, :], train_df2.loc[train_df_index]['target'])
            _oof[idx1[test_df_index]] = clf.predict_proba(train_df3[test_df_index, :])[:, 1]
            _preds[idx2] += clf.predict_proba(test_df3)[:, 1] / skf.n_splits
    
        if i % 100 == 0:
            print('Fold %d' % i)
            
    return _preds


clf1 = SVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=SEED)
preds = execute_classifier(clf1)
sub['target'] = preds / 3

clf2 = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=SEED)
preds = execute_classifier(clf2)
sub['target'] += preds / 3

clf3 = KNeighborsClassifier(n_neighbors=11, weights='distance')
preds = execute_classifier(clf3)
sub['target'] += preds / 3


sub.to_csv('submission.csv', index=False)