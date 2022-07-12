import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,VotingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix

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

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# split data into train and test
test_id = test.ID
test = test.drop(["ID"],axis=1)

X = train.drop(["TARGET","ID"],axis=1)
y = train.TARGET.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)
print(X_train.shape, X_test.shape, test.shape)

'''print  "Feature selection"
clf = ExtraTreesClassifier(random_state=1729)
selector = clf.fit(X_train, y_train)
# clf.feature_importances_ 
fs = SelectFromModel(selector, prefit=True)

X_train = fs.transform(X_train)
X_test = fs.transform(X_test)
test = fs.transform(test)

print(X_train.shape, X_test.shape, test.shape)
'''

## # Train Model
# classifier from xgboost
clf1 = AdaBoostClassifier(n_estimators=500)
clf2 = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='gini',max_depth=5)
clf3 = xgb.XGBClassifier(n_estimators=500, nthread=-1, max_depth = 5, seed=1729)
clf4 = GradientBoostingClassifier(n_estimators=500)
eclf = VotingClassifier(estimators=[('ab', clf1), ('etc', clf2), ('xgb', clf3),('gbc', clf4)], weights=[1,1,1,1], voting='soft')
eclf = eclf.fit(X_train, y_train)
# calculate the auc score
print("Roc AUC: ", roc_auc_score(y_test, eclf.predict_proba(X_test)[:,1],
              average='macro'))
              
## # Submission
probs = eclf.predict_proba(test)

submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)



