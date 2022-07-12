# Using Xian Xu's script for PCA and then using other inputs..look into the below link
# https://www.kaggle.com/godbless/santander-customer-satisfaction/feature-selection-1

import pandas as pd
import numpy as np
from numpy import mean,std,max,min

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from numpy import mean,std,max,min

import xgboost as xgb

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_final = pd.read_csv("../input/test.csv")


# remove constant columns (std = 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# split data into train and test
test_id = test.ID
#test = test.drop(["ID"],axis=1)

#Lets include the ID as well
X = train.drop(["TARGET"],axis=1)
y = train.TARGET.values

features = train.columns[1:-1]

#Setting up min-max limits

for f in features:
    lim = min(train[f])
    test.loc[test[f] < lim, f] = lim
    lim = max(train[f])
    test.loc[test[f] > lim, f] = lim
# pca
pca = PCA(n_components=3)
x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
x_test_projected = pca.transform(normalize(test[features], axis=0))
print(pca.explained_variance_ratio_)
X.insert(1, 'PCAOne', x_train_projected[:, 0])
X.insert(1, 'PCATwo', x_train_projected[:, 1])
X.insert(1, 'PCAThree', x_train_projected[:, 2])



test.insert(1, 'PCAOne', x_test_projected[:, 0])
test.insert(1, 'PCATwo', x_test_projected[:, 1])
test.insert(1, 'PCAThree', x_test_projected[:, 2])





clf = ExtraTreesClassifier(random_state=1729,bootstrap =True,class_weight = "balanced")
selector = clf.fit(normalize(X), y)
# clf.feature_importances_
fs = SelectFromModel(selector, prefit=True)

X = fs.transform(X)
test = fs.transform(test)

print(X.shape,  test.shape)



#m2_xgb = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth = 4, \
#seed=1729)
m2_xgb = xgb.XGBClassifier(missing=np.nan, max_depth=5, 
n_estimators=560, learning_rate=0.0202048, nthread=4, subsample=0.6815,
colsample_bytree=0.701, seed=4242)
metLearn = CalibratedClassifierCV(m2_xgb, method='isotonic', cv=10)
metLearn.fit(X,y)

# training Auc Score
print(roc_auc_score(y,metLearn.predict_proba(X)[:,1]))

## # Submission
probs = metLearn.predict_proba(test)
test_final['TARGET'] = probs[:,1]
test_final['nv'] = test_final['num_var33']+test_final['saldo_medio_var33_ult3']+test_final['saldo_medio_var44_hace2']+test_final['saldo_medio_var44_hace3']+test_final['saldo_medio_var33_ult1']+test_final['saldo_medio_var44_ult1']
test_final.loc[test_final.nv > 0, 'TARGET'] =0
test_final.loc[test_final.var15 < 23, 'TARGET'] = 0
test_final.loc[test_final.saldo_medio_var5_hace2 > 160000, 'TARGET'] = 0
test_final.loc[test_final.saldo_var33 > 0, 'TARGET'] = 0
test_final.loc[test_final.var38 > 3988596, 'TARGET'] = 0
test_final.loc[test_final.var21 > 7500, 'TARGET'] = 0
test_final.loc[test_final.num_var30 > 9, 'TARGET'] = 0
test_final.loc[test_final.num_var13_0 > 6, 'TARGET'] = 0
test_final.loc[test_final.num_var33_0 > 0, 'TARGET'] = 0
test_final.loc[test_final.imp_ent_var16_ult1 > 51003, 'TARGET'] = 0
test_final.loc[test_final.imp_op_var39_comer_ult3 > 13184, 'TARGET'] = 0
test_final.loc[test_final.saldo_medio_var5_ult3 > 108251, 'TARGET'] = 0
test_final.loc[(test_final['var15']+test_final['num_var45_hace3']+test_final['num_var45_ult3']+test_final['var36']) <= 24, 'TARGET'] = 0
test_final.loc[test_final.saldo_var5 > 137615, 'TARGET'] = 0
test_final.loc[test_final.saldo_var14 > 19053.78, 'TARGET'] = 0
test_final.loc[test_final.saldo_var17> 288188.97, 'TARGET'] = 0
test_final.loc[test_final.saldo_var26 > 10381.29, 'TARGET'] = 0

# Lets try some hard coding
#test_final.loc[test_final['TARGET'] < 0.0002, 'TARGET'] = 0
submission = pd.DataFrame({"ID":test_id, "TARGET": test_final['TARGET']})
submission.to_csv("submission.csv", index=False)