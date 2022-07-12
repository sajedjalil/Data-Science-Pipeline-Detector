import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

train=pd.read_csv('../input/train.csv',header=0)
testpd=pd.read_csv('../input/test.csv',header=0)
y_train = train['TARGET'].values
X_train = train.drop(['ID','TARGET'], axis=1).values
X_test = testpd.drop(['ID'], axis=1).values
rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=1)

scores = cross_validation.cross_val_score(rf, X_train, y_train, scoring='roc_auc', cv=5) 
print(scores.mean())

rf.fit(X_train,y_train)
pred = rf.predict_proba(X_test)
submission2 = pd.DataFrame({"ID":testpd.ID, "TARGET":pred[:,1]})
submission2.to_csv("submission.csv", index=False)