import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
np.random.seed(42)

train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

#params = {'C':[100, 1000], 'tol': [0.001, 0.0001]}
#log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
'''
param_grid = { "n_estimators"      : [250, 300],
           "criterion"         : ["gini", "entropy"],
           "max_features"      : [10,20],
           "max_depth"         : [10, 20],
           "min_samples_split" : [2, 4] ,
           "bootstrap": [True, False]}
#rfc = RandomForestClassifier(random_state=30)
'''
rfc = SVC(probability=True)
param_grid = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 0.05, 0.025]}
clf = GridSearchCV(rfc, param_grid, scoring='neg_log_loss', refit='True', n_jobs=-1, cv=5)
clf.fit(x_train, y_train)

print("best params: " + str(clf.best_params_))
for params, mean_score, scores in clf.grid_scores_:
  print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))
  print(scores)

test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
x_test = scaler.transform(x_test)

y_test = clf.predict_proba(x_test)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission_log_reg.csv')