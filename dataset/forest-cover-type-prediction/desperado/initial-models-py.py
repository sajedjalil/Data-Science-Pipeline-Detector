import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

print("Description of training set")
print(train.describe())

# Training data
X_train = train.drop(['Cover_Type', 'Id'], axis=1)
y = train[['Cover_Type']]
# Test data
X_test = test.drop(['Id'], axis=1)

# Any files you write to the current directory get shown as outputs
from sklearn.ensemble import ExtraTreesClassifier

model_et = ExtraTreesClassifier(n_estimators=200)
model_et = model_et.fit(X_train, y)
print(model_et.score(X_train, y))

predictions = model_et.predict(X_test)
print(predictions)

c1 = pd.DataFrame(test["Id"])
c2 = pd.DataFrame({'Cover_Type' : predictions})
res = (pd.concat([c1,c2], axis=1))
res.to_csv('ExtraTreesClassifier.csv', index=False)
