import pandas as pd
from sklearn import ensemble

# The competition datafiles are in the directory ../input
# Read competition data files:
# train = pd.read_csv("../input/train.csv")
# test  = pd.read_csv("../input/test.csv")

# Write to the log:
# print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# print("Description of training set")
# print(train.describe())


# Any files you write to the current directory get shown as outputs
loc_train = "train.csv"
loc_test = "test.csv"
loc_submission = "kaggle.forest.submission.150.csv"

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]
y = df_train['Cover_Type']
test_ids = df_test['Id']

clf = ensemble.RandomForestClassifier(n_estimators = 150)

clf.fit(X_train, y)
with open(loc_submission, "wb") as outfile:
    outfile.write(bytes("Id,Cover_Type\n", 'UTF-8'))
    for e, val in enumerate(list(clf.predict(X_test))):
      outfile.write(bytes("%s,%s\n"%(test_ids[e],val), 'UTF-8'))