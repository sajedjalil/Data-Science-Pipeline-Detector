import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import linear_model, metrics
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# clean and split data
print ("dimention of the traing data"+ str(train.shape))
print ("dimention of the test data"+ str(train.shape))


# remove constant columns (std = 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)


train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
print ("removing " + str(len(remove))+ "vars")
print ("dimention of the traing removing 0 sd"+ str(train.shape))
print ("dimention of the test removing 0 sd"+ str(train.shape))



# remove duplicated columns
remove_dups = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove_dups.append(cols[j])

print ("removing " + str(len(remove_dups))+ "vars")
train.drop(remove_dups, axis=1, inplace=True)
test.drop(remove_dups, axis=1, inplace=True)


print ("dimention of the traing data after duplicated "+ str(train.shape))
print ("dimention of the test data after duplicated "+ str(train.shape))



# split data into train and test
test_id = test.ID
test = test.drop(["ID"],axis=1)




X = train.drop(["TARGET","ID"],axis=1)
y = train.TARGET.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1729)


print(X_train.shape, X_test.shape, test.shape)

## # Feature selection
clf = RandomForestClassifier(random_state=1729)
selector = clf.fit(X_train, y_train)
# clf.feature_importances_ 
fs = SelectFromModel(selector, prefit=True)

X_train = fs.transform(X_train)
X_test = fs.transform(X_test)
test = fs.transform(test)

print(X_train.shape, X_test.shape, test.shape)


## # Train Model
# classifier from xgboost
m2_xgb = xgb.XGBClassifier(n_estimators=110, nthread=-1, seed=1729)
m2_xgb.fit(X_train, y_train, eval_metric="auc",
           eval_set=[(X_test, y_test)])

# calculate the auc score
print("Roc AUC: ", roc_auc_score(y_test, m2_xgb.predict_proba(X_test)[:,1],
              average='macro'))
              
## # Submission
probs = m2_xgb.predict_proba(test)

submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)



