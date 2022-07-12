# thanks to https://www.kaggle.com/speedwagon/quadratic-discriminant-analysis
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

# read in and split data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

drop = ["id", "target", "wheezy-copper-turtle-magic"]
feature_cols = [ c for c in train.columns if c not in drop ]

skf = StratifiedKFold(n_splits=21, random_state=42)
clf = KNeighborsClassifier(5)

# prep result dataframe
sub = test[["id"]].copy()
sub["target"] = None
num_sets = train['wheezy-copper-turtle-magic'].max() + 1

train_preds = np.zeros(train.shape[0])
preds = np.zeros(test.shape[0])

for i in range(num_sets):
    train_data = train[train['wheezy-copper-turtle-magic'] == i]
    test_data = test[test['wheezy-copper-turtle-magic'] == i]
    
    data = pd.concat([train_data[feature_cols], test_data[feature_cols]])

    vt = VarianceThreshold(threshold=1.5).fit(data)
    
    slim_train_features = vt.transform(train_data[feature_cols])
    slim_test_features = vt.transform(test_data[feature_cols])

    for train_index, test_index in skf.split(slim_train_features, train_data['target']):
        clf.fit(slim_train_features[train_index, :], train_data.iloc[train_index]['target'])
        train_preds[train_data.index[test_index]] += clf.predict_proba(slim_train_features[test_index, :])[:, 1]
        preds[test_data.index] += clf.predict_proba(slim_test_features)[:, 1] / skf.n_splits

print(roc_auc_score(train['target'], train_preds))
sub["target"] = preds
sub[["id", "target"]].to_csv("knn_submission_v3.csv", index=False)