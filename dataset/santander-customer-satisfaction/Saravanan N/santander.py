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
                                                                            test_size=0.3)
dtrain = xgb.DMatrix(features_train.as_matrix(), label=labels_train)
dtest = xgb.DMatrix(features_test.as_matrix(), label=labels_test)
watch_list = [(dtrain, 'train'), (dtest, 'test')]
params = {'booster':'gbtree', 'bst:max_depth':3, 'bst:eta':0.05, 'bst:alpha':0.5, 'silent':1, 'objective':'binary:logistic',
         "eval_metric":"auc", "colsample_bytree":0.7, "subsample":0.7}
clf = xgb.train(params, dtrain, num_boost_round=10, evals=watch_list, maximize=False)
pred = clf.predict(dtest)
pred_1_train = clf.predict(dtrain)
pred_1 = pred

from sklearn.metrics import roc_auc_score
print(roc_auc_score(labels_test, pred))

#AdaBoost Classifer
from sklearn.ensemble import AdaBoostClassifier

clf_2 = AdaBoostClassifier()
clf_2.fit(features_train, labels_train)
pred = clf_2.predict_proba(features_test)
pred_2_train = clf_2.predict_proba(features_train)[:,1]
pred_2 = pred[:,1]
print(roc_auc_score(labels_test, pred[:,1]))

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
old_selected_features = selected_features
selected_features = [
    "var15",
    "saldo_var5",
    "saldo_var30",
    "saldo_var42",
    "var36",
    "num_var22_hace2",
    "num_var22_hace3",
    "num_var22_ult3",
    "num_var45_hace2",
    "num_var45_hace3",
    "num_var45_ult3",
    "saldo_medio_var5_hace2",
    "saldo_medio_var5_hace3",
    "saldo_medio_var5_ult1",
    "saldo_medio_var5_ult3",
    "var38",
]
selected_features_ = selected_features
old_features_train = features_train
features_train = features_train.ix[:,selected_features]
#clf_3 = GaussianNB()
#clf_3 = LogisticRegression()
clf_3 = GradientBoostingClassifier()
clf_3.fit(features_train, labels_train)
pred = clf_3.predict_proba(features_test.ix[:,selected_features_])
pred_3_train = clf_3.predict_proba(features_train)[:,1]
pred_3 = pred[:,1]
print(roc_auc_score(labels_test, pred[:,1]))
selected_features = old_selected_features


final_clf = GaussianNB()
final_clf.fit(np.column_stack([pred_1_train, pred_3_train]), labels_train)
average = final_clf.predict_proba(np.column_stack([pred_1, pred_3]))[:,1]
#print(roc_auc_score(labels_test, pred[:,1]))


print(roc_auc_score(labels_test, average))
# test_data = pd.read_csv("../input/test.csv")
# features = test_data.ix[:,selected_features]#.as_matrix()
# ids = test_data.ix[:, "ID"]
# test_features = xgb.DMatrix(features)
# pred_1 = clf.predict(test_features)
# pred_2 = clf_2.predict_proba(features)[:,1]
# pred_3 = clf_3.predict_proba(features.ix[:,selected_features_])[:,1]
# pred = (pred_1*1.5 + pred_2 + pred_3*0.5)/4
# output_data = pd.concat([ids, pd.DataFrame(pred, columns=["TARGET"])], axis=1)
# output_data.to_csv("output.csv", index=False)
