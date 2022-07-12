import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from itertools import combinations
from numpy import array,array_equal

from sklearn import cross_validation as cv
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn import ensemble
from sklearn import linear_model 
from sklearn import naive_bayes 

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import xgboost as xgb


train_dataset = pd.read_csv("../input/train.csv", index_col=0)
test_dataset = pd.read_csv("../input/test.csv", index_col=0)

print('Train: {}\nTest: {}'.format(train_dataset.shape, test_dataset.shape))


#####################
# Remove constant features
def identify_constant_features(dataframe):
    count_uniques = dataframe.apply(lambda x: len(x.unique()))
    constants = count_uniques[count_uniques == 1].index.tolist()
    return constants

constant_features_train = set(identify_constant_features(train_dataset))
print('There were {} constant features in TRAIN dataset.'.format(
        len(constant_features_train)))
        
# Drop constant features
train_dataset.drop(constant_features_train, inplace=True, axis=1)
test_dataset.drop(constant_features_train, inplace=True, axis=1)
print('Train: {}\nTest: {}'.format(train_dataset.shape, test_dataset.shape))


#######################
# Remove equal features
def identify_equal_features(dataframe):
    features_to_compare = list(combinations(dataframe.columns.tolist(),2))
    equal_features = []
    for compare in features_to_compare:
        is_equal = array_equal(dataframe[compare[0]],dataframe[compare[1]])
        if is_equal:
            equal_features.append(list(compare))
    return equal_features

equal_features_train = identify_equal_features(train_dataset)
print('There were {} pairs of equal features in TRAIN dataset.'.format(len(equal_features_train)))

# Remove the second feature of each pair.
features_to_drop = array(equal_features_train)[:,1] 
train_dataset.drop(features_to_drop, axis=1, inplace=True)
test_dataset.drop(features_to_drop, axis=1, inplace=True)
print('Train: {}\nTest: {}'.format(train_dataset.shape, test_dataset.shape))


#####################
# Remove correlated features
def identify_corr_features(dataframe):
    features_to_compare = list(combinations(dataframe.columns.tolist(),2))
    corr_features = []
    for compare in features_to_compare:
        corr = np.corrcoef(dataframe[compare[0]],dataframe[compare[1]])[0,1]
        if corr > 0.99 :
            corr_features.append(list(compare))
    return corr_features

#highCorr_features_train = identify_corr_features(train_dataset)
#print('There were {} pairs of equal correlation in TRAIN dataset.'.format(len(highCorr_features_train)))
#print('Pairs of equal correlation in TRAIN dataset{}.'.format(highCorr_features_train))

# Remove the second feature of each pair.
#features_to_drop = array(highCorr_features_train)[:,1] 
#train_dataset.drop(features_to_drop, axis=1, inplace=True)
#test_dataset.drop(features_to_drop, axis=1, inplace=True)
#print('Train: {}\nTest: {}'.format(train_dataset.shape, test_dataset.shape))


###################
# Define variables for model.
y_name = 'TARGET'
feature_names = train_dataset.columns.tolist()
feature_names.remove(y_name)

X_train = train_dataset[feature_names]
y_train = train_dataset[y_name]
#X_train = X_train.replace(-999999,2)

X_test = test_dataset[feature_names]
#X_test = X_test.replace(-999999,2)

id_test = X_test.index
#################
# classifier
# gave lb score of .839 and Rank of 1071
#[374]	validation_0-auc:0.892980
#Overall AUC: 0.884567960939
#clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=375, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
# gave lb score of  and Rank of
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=560, learning_rate=0.02, nthread=4, subsample=0.7, colsample_bytree=0.7, seed=400)
X_fit, X_eval, y_fit, y_eval= cv.train_test_split(X_train, y_train, test_size=0.3)

# fitting
clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])

print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

# predicting
y_pred= clf.predict_proba(X_test)[:,1]

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)

print('Completed!')


