import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

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

#split
test_id = test.ID
test = test.drop(["ID"],axis=1)

X = train.drop(["TARGET","ID"],axis=1)
y = train.TARGET.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(X_train.shape, X_test.shape, test.shape)

#Feature selection
sclf = ExtraTreesClassifier(n_estimators=47,max_depth=47)
selector = sclf.fit(X_train, y_train)
fs = SelectFromModel(selector, prefit=True)

X_train = fs.transform(X_train)
X_test = fs.transform(X_test)
test = fs.transform(test)

print(X_train.shape, X_test.shape, test.shape)

#loop
names = ["etsc","dtc","etc","abc","xgb","gbc"]
clfs = [
ExtraTreesClassifier(n_estimators=100,max_depth=5),
DecisionTreeClassifier(max_depth=5),
ExtraTreeClassifier(max_depth=5),
AdaBoostClassifier(n_estimators=100),
xgb.XGBClassifier(n_estimators=100, nthread=-1, max_depth = 5),
GradientBoostingClassifier(n_estimators=100,max_depth=5)
]

plt.figure()
for name,clf in zip(names,clfs):

	clf.fit(X_train,y_train)
	y_proba = clf.predict_proba(X_test)[:,1]
	print("Roc AUC:"+name, roc_auc_score(y_test, clf.predict_proba(X_test)[:,1],average='macro'))
	fpr, tpr, thresholds = roc_curve(y_test, y_proba)
	plt.plot(fpr, tpr, label=name)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('1.png')
plt.show()          

#probs = xgb.predict_proba(test)
#submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
#submission.to_csv("submission.csv", index=False)



