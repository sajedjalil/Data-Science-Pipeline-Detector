import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

#params = {'C':[1, 10, 50, 100, 500, 1000, 2000], 'tol': [0.001, 0.0001, 0.005], 'solver':  ["newton-cg"]}
#log_reg = LogisticRegression(multi_class="multinomial")
#clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=1, cv=5)
#clf.fit(x_train, y_train)
#y_test = clf.predict_proba(x_test)

log_reg = LogisticRegression(C=2000, multi_class="multinomial", tol=0.0001, solver='newton-cg')
log_reg.fit(x_train, y_train)
y_test = log_reg.predict_proba(x_test)


#params = {'n_estimators':[1, 10, 50, 100, 500]}
#random_forest = RandomForestClassifier()
#clf = GridSearchCV(random_forest, params, scoring='log_loss', refit='True', n_jobs=1, cv=5)
#clf.fit(x_train, y_train)
#y_test = clf.predict_proba(x_test)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission2.csv')