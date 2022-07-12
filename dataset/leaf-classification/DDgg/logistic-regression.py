import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


np.random.seed(12345)

train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

# possible param options
# multiclass: ‘newton-cg’, ‘sag’ and ‘lbfgs’ 
# max_iter : int, default: 100
# tol: Tolerance for stopping criteria.


params = {'C':[1800,2000,2200], 'tol': [0.0001]}
log_reg = LogisticRegression(solver='sag', multi_class='multinomial',
class_weight='balanced',max_iter=400)


clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=-1, cv=5)
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