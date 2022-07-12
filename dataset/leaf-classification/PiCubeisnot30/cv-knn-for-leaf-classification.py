import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.grid_search import GridSearchCV


train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values


knn = neighbors.KNeighborsClassifier(weights='uniform')
params = {'n_neighbors':[1,2,3,4,5,5,6,7]}
cv_knn= GridSearchCV(knn,params, scoring='log_loss', refit='False', n_jobs=-1, cv=5)
cv_knn.fit(x_train, y_train)
y_test = cv_knn.predict_proba(x_test)


submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission4.csv')
