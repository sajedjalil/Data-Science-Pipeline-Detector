import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import normalize
from sklearn.decomposition import SparsePCA

import xgboost as xgb

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# clean and split data

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


print('following features were removed')
print('total removed features = '+ str(len(remove)))

print(remove)


train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# split data into train and test
test_id = test.ID
test = test.drop(["ID"],axis=1)

X = train.drop(["TARGET","ID"],axis=1)
y = train.TARGET.values

features = train.columns[1:-1]

# pca
pca = SparsePCA(n_components=3)
x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
x_test_projected = pca.transform(normalize(test[features], axis=0))
#print(pca.explained_variance_ratio_)
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
m2_xgb = xgb.XGBClassifier(missing=np.nan, max_depth=6, 
n_estimators=350, learning_rate=0.025, nthread=4, subsample=0.95,
colsample_bytree=0.85, seed=4242)
metLearn = CalibratedClassifierCV(m2_xgb, method='isotonic', cv=10)
metLearn.fit(X,y)

print ('done')
## # Submission
probs = metLearn.predict_proba(test)

submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)





