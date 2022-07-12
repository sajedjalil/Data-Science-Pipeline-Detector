import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import scale

print('Reading data...')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print('Fitting KNN...')
# X = train[['x', 'y', 'accuracy', 'time']]
X = train[['x', 'y']]
y = train['place_id'].astype('category')

# scale data
# X = pd.DataFrame(scale(X))

# use sqrt(n) neighbors
knn = KNeighborsClassifier(n_neighbors = 1, n_jobs=-1)
knn.fit(X,y)

print('Predicting results for 100 check-ins...')
test_id = test.loc[:99, 'row_id']
X_new = test.loc[:99, ['x', 'y']]

preds = knn.predict(X_new)

print('Writing to csv...')
submission = pd.DataFrame({'row_id': test_id, 'place_id': preds}, columns = ['row_id', 'place_id'])
submission.to_csv('submission.csv', index=False)
