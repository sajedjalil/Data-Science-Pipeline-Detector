import pandas as pd
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
submission = "kaggle.rf200.entropy.submission.csv"

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

#print("Description of training set")
#print(train.describe())

# Any files you write to the current directory get shown as outputs
feature_cols = [col for col in train.columns if col not in ['Cover_Type','Id']]
cols_to_norm = ["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Hillshade_9am",
"Horizontal_Distance_To_Roadways","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"]

#train[cols_to_norm] = train[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#test[cols_to_norm] = test[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

X_train = train[feature_cols]
X_test = test[feature_cols]
y = train['Cover_Type']
test_ids = test['Id']
del train
del test

clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X_train, y)

del X_train

with open(submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))