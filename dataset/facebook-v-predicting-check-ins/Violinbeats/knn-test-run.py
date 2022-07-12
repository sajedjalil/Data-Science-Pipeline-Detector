import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

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
# knn = KNeighborsClassifier(n_neighbors= round(X.shape[0] ** 0.5), weights='distance', n_jobs=-1)

# use 3 neighbors
knn = KNeighborsClassifier(n_neighbors= 10, weights='distance', n_jobs=-1)
knn.fit(X,y)

print('Predicting results on 100 new check-ins...')
test_id = test.loc[:99, 'row_id']
X_new = test.loc[:99, ['x', 'y']]
preds = knn.predict_proba(X_new)

yCols = y.unique()
def probs2Places(preds, yCols):
    '''Convert probabilities to top 3 place_ids'''
    results = [list(yCols[pred.argsort()[-3:][::-1]]) for pred in preds]
    return ['{0} {1} {2}'.format(result[0], result[1], result[2]) for result in results]

place_id = probs2Places(preds, yCols)

print('Writing to csv...')
submission = pd.DataFrame({'row_id': test_id, 'place_id': place_id}, columns=['row_id', 'place_id'])
submission.to_csv('submission.csv', index=False)
