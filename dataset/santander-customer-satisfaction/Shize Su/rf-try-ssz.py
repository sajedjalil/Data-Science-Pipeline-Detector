import pandas
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np
np.random.seed(88)

df_train = pandas.read_csv("../input/train.csv")
df_test  = pandas.read_csv("../input/test.csv")   



id_test = df_test['ID']
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values
X_test = df_test.drop(['ID'], axis=1).values

clf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=8)

scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=5) 
print(scores.mean())


clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

submission = pandas.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("z_rf_try4_ssz.csv", index=False)

