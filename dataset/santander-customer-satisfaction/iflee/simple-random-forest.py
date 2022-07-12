import pandas
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

df_train = pandas.read_csv("../input/train.csv")
df_test  = pandas.read_csv("../input/test.csv")   

id_test = df_test['ID']
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values
X_test = df_test.drop(['ID'], axis=1).values

clf = RandomForestClassifier(n_estimators=100, max_depth=17, random_state=1)

scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=5) 
print(scores.mean())


clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

submission = pandas.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

