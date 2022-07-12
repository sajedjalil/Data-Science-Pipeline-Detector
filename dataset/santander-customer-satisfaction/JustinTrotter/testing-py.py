from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# remove constant columns (standard deviation == 0)
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
test = test.drop(["ID"],axis=1)
X = train.drop(["TARGET","ID"],axis=1)
y = train.TARGET.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00375, random_state=3542)

# feature selection
clf = DecisionTreeClassifier(random_state=7541, max_features="log2", max_depth=None, min_samples_split=1)
selector = clf.fit(X_train, y_train)

# clf.feature_importances_ 
fs = SelectFromModel(selector, prefit=True)

X_train = fs.transform(X_train)
X_test = fs.transform(X_test)
test = fs.transform(test)

# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=6744)
# fitting
clf.fit(X_train, y_train, eval_metric="auc", eval_set=[(X_test, y_test)])
#run classifier

# submission
probs = clf.predict_proba(test)              

submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)



