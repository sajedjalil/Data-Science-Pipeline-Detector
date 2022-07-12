import numpy as np

# Load the data
data = np.genfromtxt('../input/train.csv', delimiter=',')[1:]
X = data[:,2:]
Y = data[:,1]

print('Loaded', len(X), 'samples')
print('Maps', X.shape[1], 'dimensional data to one-dimensional data')

from sklearn.decomposition import PCA

def perform_pca(X, k):
    """ Performs a PCA to reduce to k dimensions
    X - The dataset
    k - The dimension of the result
    """
    pca = PCA(n_components=k)
    A = pca.fit_transform(X)

    return A, pca

# The dimensions to be used
dims = [5, 10, 20, 30, 50, 75]

# Perform a k-fold validation split
from sklearn.model_selection import KFold

kfold = KFold(n_splits=len(dims), shuffle=True)

# We will train a series of linear regression models
from sklearn.linear_model import LogisticRegression

best = None

for train, test in kfold.split(data):
    # Perform the split; this gives us our data
    X_train = X[train]
    Y_train = Y[train]
    X_test = X[test]
    Y_test = Y[test]

    # Get the dimension
    d = dims.pop()
    print('Will generate PCA of dimension', d)

    # Perform PCA
    X_train, pca = perform_pca(X_train, d)
    X_test = pca.transform(X_test)

    # Now, we can train a linear regression model on the data
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    
    # Evaluate the model
    score = clf.score(X_test, Y_test)
    
    print('Score:', score)

    if not best or score > best[-1]:
        # Save the PCA method, the computer, and the score
        best = pca, clf, score

# Grab the best model
pca, clf, score = best

# Load the test data
data = np.genfromtxt('../input/test.csv', delimiter=',')[1:]
X = pca.transform(data[:,1:])

# Make predictions
Y = clf.predict(X)

# Save to a file
import csv
with open('output.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['id', 'target'])
    writer.writeheader()
    writer.writerows([{'id': 250+i, 'target': 1 if Y[i] > 0 else 0} for i in range(len(X))])