import pandas as pd
import sklearn
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

# Import Data
X = pd.read_csv('../input/train.csv')
X = X.drop('id', axis=1)

# Extract target
# Encode it to make it manageable by ML algo
y = X.target.values
y = LabelEncoder().fit_transform(y)

# Remove target from train, else it's too easy ...
X = X.drop('target', axis=1)

X = X.applymap(math.log1p)

# Split Train / Test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.8, random_state=42)

clf = LogisticRegression(C=1e10)
clf.fit(Xtrain, ytrain)
ypreds = clf.predict_proba(Xtest)
print("training loss: ", log_loss(ytrain, clf.predict_proba(Xtrain), eps=1e-15, normalize=True))
print("validation loss: ", log_loss(ytest, clf.predict_proba(Xtest), eps=1e-15, normalize=True))