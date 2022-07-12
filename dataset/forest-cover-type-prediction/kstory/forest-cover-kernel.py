import numpy as np
from numpy import genfromtxt
import xgboost as xgb
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# The competition datafiles are in the directory ../input
# Read competition data files:
train = genfromtxt("../input/train.csv", delimiter=',')
test = genfromtxt("../input/test.csv", delimiter=',')
sample_sub = genfromtxt("../input/sampleSubmission.csv", delimiter=',')
print("Data loaded.")

# print(train[~np.isnan(train).any(axis=1)])

train = train[~np.isnan(train).any(axis=1)]
test = test[~np.isnan(test).any(axis=1)]
sample_sub = sample_sub[~np.isnan(sample_sub).any(axis=1)]

train_feat = train[:, 1:-1]
test_feat = test[:, 1:]
features = np.append(train_feat, test_feat, axis=0)

train_label = train[:, -1]
test_label = sample_sub[0:test_feat.shape[0], 1]
print(train_label.shape)
print(test_label.shape)
labels = np.append(train_label, test_label, axis=0)

print(features.shape)
print(labels.shape)


# Write to the log:
# print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# print("Description of training set")
# print(train.describe())

# scaler = StandardScaler()
# scaler.fit(train_feat)
# train_feat = scaler.transform(train_feat)
# test_feat = scaler.transform(test_feat)

X_train, X_cv, y_train, y_cv = train_test_split(features, labels, test_size=0.2)
X_test = test_feat
y_test = test_label

# pca = PCA(n_components=5)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)

model = xgb.XGBClassifier()
print("Fitting model...")
model.fit(X_train, y_train)
print("Finished.")

pred = model.predict(X_cv)
print("Prediction ready.")
print(y_cv.shape)
print(pred.shape)
acc = sklearn.metrics.accuracy_score(y_cv, pred)

'''
for i in range(y_cv.shape[0]):
    if y_cv[i] != pred[i]:
        print(X_cv[i])
        print(y_cv[i], pred[i])
'''

print("CV Accuracy:", acc)

