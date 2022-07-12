# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools

train = pd.read_csv("../input/train.csv", header=0, sep=',', index_col=0)
test = pd.read_csv("../input/test.csv", header=0, sep=',', index_col=0)

drop_list = ['cat15','cat18','cat20','cat21','cat22','cat48','cat55','cat56','cat58','cat59','cat60'
             ,'cat62','cat63','cat64','cat65','cat68','cat69','cat77','cat78','cat85']

'''
def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain
'''

train = train.drop(drop_list,axis=1)
test = test.drop(drop_list,axis=1)

'''
bins1 = [x for x in np.arange(0,1.2,0.2)]
train.cont1 = pd.cut(train.cont1,bins1)
test.cont1 = pd.cut(test.cont1,bins1)
bins2 = [x for x in np.arange(0,1.05,0.05)]
train.cont2 = pd.cut(train.cont2,bins2)
test.cont2 = pd.cut(test.cont2,bins2)

train.rename(columns={"cont1": "cat001", "cont2": "cat002"},inplace = True)
test.rename(columns={"cont1": "cat001", "cont2": "cat002"},inplace = True)

numeric_feats = [x for x in train.columns[0:-1] if 'cont' in x]


train_test, ntrain = mungeskewed(train, test, numeric_feats)


train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))
test.cont7 = np.log1p(test.cont7)
train_test["cont6"] = np.log1p(preprocessing.minmax_scale(train_test["cont6"]))
train_test["cont7"] = np.log1p(preprocessing.minmax_scale(train_test["cont7"]) )
train_test["cont9"] = np.log1p(preprocessing.minmax_scale(train_test["cont9"]) )
train_test["cont13"] = np.log1p(preprocessing.minmax_scale(train_test["cont13"]) )
#train_test["cont14"] = (np.maximum(train_test["cont14"] - 0.179722, 0) / 0.665122) ** 0.25


ss = StandardScaler()
train_test[numeric_feats] = ss.fit_transform(train_test[numeric_feats].values)

train = train_test.iloc[:ntrain, :].copy()
test = train_test.iloc[ntrain:, :].copy()
'''

cat_features = [x for x in train.columns[0:-1] if 'cat' in x]

for c in range(len(cat_features)):
    train[cat_features[c]] = train[cat_features[c]].astype('category').cat.codes   
    test[cat_features[c]] = test[cat_features[c]].astype('category').cat.codes  


train['loss'] = np.log(train['loss'])
#test = test.drop(["loss"],axis = 1)


X_train = train.drop("loss",axis=1)
Y_train = train['loss']
X_test = test


print('Data Prepared')


#model = RandomForestRegressor(criterion='mse', n_jobs = -1, n_estimators = 300, max_depth=None,
#min_samples_split=2, min_samples_leaf=1,
#min_weight_fraction_leaf=0.0, max_features='auto',
#max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True,
#oob_score=False, random_state=None, verbose=0, warm_start=False)

RF_model = RandomForestRegressor(criterion='mse', n_jobs = -1, oob_score = True )

parameters = {'max_features':[80,'auto'], 
              'n_estimators':[50],
              'min_samples_leaf': [1,50]}

def _score_func(estimator, X, y):
    return mean_absolute_error(y, estimator.predict(X))


clf = GridSearchCV(RF_model, parameters, 
                   cv=KFold(len(Y_train), n_folds = 3, shuffle = True), 
                   scoring=_score_func,
                   verbose=2, refit=True)

clf.fit(X_train, Y_train)
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])

print(best_parameters)
print(score)
print('Model ran')

print(clf.score(X_train, Y_train))

y_predict = clf.predict(X_test)
y_predict = np.exp(y_predict)
submit = pd.read_csv("../input/sample_submission.csv")
submit["loss"] = y_predict
submit.to_csv("randomforest_trial.csv", index=False)

#print(model.feature_importances_)

#print(RF_model.oob_score_)
#print(RF_model.feature_importances_)

# Any results you write to the current directory are saved as output.