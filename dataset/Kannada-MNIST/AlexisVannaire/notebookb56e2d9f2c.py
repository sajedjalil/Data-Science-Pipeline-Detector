# Libraries

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC


# Importation
train_set = "../input/Kannada-MNIST/train.csv"
valid_set = "../input/Kannada-MNIST/Dig-MNIST.csv"
test_set = "../input/Kannada-MNIST/test.csv"

train = pd.read_csv(train_set)
test = pd.read_csv(test_set)

X, y = train.iloc[:,1:], train.iloc[:,0]


# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.98)
X = pca.fit_transform(X_scaled)


# Training
svm_poly = SVC(kernel="poly", coef0=0.04, degree=3, C=10**5)
svm_poly.fit(X, y)


# Predictions
x_test = test.drop("id", axis=1).values.astype('float32')
x_test = scaler.transform(x_test)
x_test = pca.transform(x_test)
pred = svm_poly.predict(x_test).astype(int)


def write_preds(preds, fname):
    pd.DataFrame({"id": list(range(0, len(preds))), "label": preds}).to_csv(fname, index=False, header=True)

write_preds(pred, "submission.csv")