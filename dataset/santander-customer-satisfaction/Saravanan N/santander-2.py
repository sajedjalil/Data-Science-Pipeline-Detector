# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import xgboost as xgb
data = pd.read_csv("../input/train.csv")

#indices = [index for index, i in data.ix[:, "var38"].iteritems() if i > 100000]
#data.drop(indices, inplace=True)

from sklearn.feature_selection import VarianceThreshold

def remove_feat_constants(data_frame):
    # from https://www.kaggle.com/tuomastik/santander-customer-satisfaction/pca-visualization
    # script by Tuomas Tikkanen
    # Remove feature vectors containing one unique value,
    # because such features do not have predictive value.
    # Let's get the zero variance features by fitting VarianceThreshold
    # selector to the data, but let's not transform the data with
    # the selector because it will also transform our Pandas data frame into
    # NumPy array and we would like to keep the Pandas data frame. Therefore,
    # let's delete the zero variance features manually.
    n_features_originally = data_frame.shape[1]
    selector = VarianceThreshold()
    selector.fit(data_frame)
    # Get the indices of zero variance feats
    feat_ix_keep = selector.get_support(indices=True)
    orig_feat_ix = np.arange(data_frame.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
    # Delete zero variance feats from the original pandas data frame
    data_frame = data_frame.drop(labels=data_frame.columns[feat_ix_delete],
                                 axis=1)
    # Print info
    n_features_deleted = feat_ix_delete.size
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame

def remove_feat_identicals(data_frame):
    # from https://www.kaggle.com/tuomastik/santander-customer-satisfaction/pca-visualization
    # script by Tuomas Tikkanen
    # Find feature vectors having the same values in the same order and
    # remove all but one of those redundant features.
    n_features_originally = data_frame.shape[1]
    # Find the names of identical features by going through all the
    # combinations of features (each pair is compared only once).
    feat_names_delete = []
    for feat_1, feat_2 in itertools.combinations(
            iterable=data_frame.columns, r=2):
        if np.array_equal(data_frame[feat_1], data_frame[feat_2]):
            feat_names_delete.append(feat_2)
    feat_names_delete = np.unique(feat_names_delete)
    # Delete the identical features
    data_frame = data_frame.drop(labels=feat_names_delete, axis=1)
    n_features_deleted = len(feat_names_delete)
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame
    
data = remove_feat_constants(data)
data = remove_feat_identicals(data)
features = data.ix[:,"var3":"var38"]
selected_features = features.columns
features = data.ix[:, selected_features]
#features = features.as_matrix()
labels = data.ix[:, "TARGET"].as_matrix()
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels, 
                                                                            test_size=0.4)
train_error = []
test_error = []
for i in range(1, int(len(features_train)/1000) + 1):
    dtrain = xgb.DMatrix(features_train.iloc[0:i * 1000].as_matrix(), label=labels_train[:i * 1000])
    dtest = xgb.DMatrix(features_test.as_matrix(), label=labels_test)
    #watch_list = [(dtrain, 'train'), (dtest, 'test')]
    params = {'booster':'gbtree', 'bst:max_depth':5, 'bst:eta':0.05, 'bst:lambda':0.9, 'silent':1, 'objective':'binary:logistic',
        "eval_metric":"auc", "colsample_bytree":0.7, "subsample":0.7}
    #evals=watch_list,     
    clf = xgb.train(params, dtrain, num_boost_round=50, maximize=False)
    pred = clf.predict(dtest)
    pred_1 = clf.predict(dtrain)
    from sklearn.metrics import roc_auc_score
    try:
        print(roc_auc_score(labels_test, pred))
        print(roc_auc_score(labels_train[:i * 1000], pred_1))
        train_error.append((i* 1000, roc_auc_score(labels_train[:i * 1000], pred_1)))
        test_error.append((i* 1000, roc_auc_score(labels_test, pred)))
        print("++++++++++++++++++++++++++++++++")
    except Exception as e:
        print(e)
        print("error")