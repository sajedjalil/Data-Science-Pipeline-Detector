import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1)
corr= x_train.corr()
remove = []
col= x_train.columns.values
for i in range(len(col)-1):
    for j in range(i+1,len(col)-1):
        if corr[col[i]][col[j]] > 0.80:
            if col[j] not in remove:
                remove.append(col[j])
x_train = x_train.drop(remove,axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.drop(remove,axis=1).values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

params = {'C':[1, 10, 50, 100, 500, 1000, 2000], 'tol': [0.001, 0.0001, 0.005]}
log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=1, cv=5)
clf.fit(x_train, y_train)

print("best params: " + str(clf.best_params_))

y_test = clf.predict_proba(x_test)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submissionLR.csv')