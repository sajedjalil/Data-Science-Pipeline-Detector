import pandas as pd
import numpy as np
import math
import zipfile
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def llfun(act, pred):
    """ Logloss function for 1/0 probability
    """
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)

z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])[['X', 'Y', 'Category']]


# Separate test and train set out of orignal train set.
msk = np.random.rand(len(train)) < 0.8
knn_train = train[msk]
knn_test = train[~msk]
n = len(knn_test)

print("Original size: %s" % len(train))
print("Train set: %s" % len(knn_train))
print("Test set: %s" % len(knn_test))

# Prepare data sets
x = knn_train[['X', 'Y']]
y = knn_train['Category'].astype('category')
actual = knn_test['Category'].astype('category')


# Fit
logloss = []
for i in range(1, 50, 1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x, y)
    
    # Predict on test set
    outcome = knn.predict(knn_test[['X', 'Y']])
    
    # Logloss
    logloss.append(llfun(actual, outcome))

plt.plot(logloss)
plt.savefig('n_neighbors_vs_logloss.png')

# Submit for K=40
z = zipfile.ZipFile('../input/test.csv.zip')
test = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'])
x_test = test[['X', 'Y']]
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(x, y)
outcomes = knn.predict(x_test)

submit = pd.DataFrame({'Id': test.Id.tolist()})
for category in y.cat.categories:
    submit[category] = np.where(outcomes == category, 1, 0)
    
submit.to_csv('k_nearest_neigbour.csv', index = False)
