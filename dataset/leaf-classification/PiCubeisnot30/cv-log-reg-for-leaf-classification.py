import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values

log_reg = LogisticRegression( solver='newton-cg',multi_class='multinomial')
params = {'C':[0.1,100,1000]}
cv_log_reg = GridSearchCV(log_reg,params, scoring='log_loss', refit='False', n_jobs=-1, cv=5)
cv_log_reg.fit(x_train, y_train)
y_test = cv_log_reg.predict_proba(x_test)

#print(y_test)
submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission4.csv')