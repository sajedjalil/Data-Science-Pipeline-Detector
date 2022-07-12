import pandas
import numpy as np

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

df_train = pandas.read_csv("../input/train.csv")
df_test  = pandas.read_csv("../input/test.csv")   

#remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)
        
df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

#remove duplicate columns
remove = []
c = df_train.columns

for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i + 1, len(c)):
        if np.array_equal(v, df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

id_test = df_test['ID']
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values
X_test = df_test.drop(['ID'], axis=1).values

clf = RandomForestClassifier(n_estimators=400, max_depth=17, random_state=1, verbose = 2)

scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=5) 
print(scores.mean())


clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

submission = pandas.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

