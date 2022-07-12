import numpy as np 
import pandas as pd 
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFpr
trainDF = pd.read_csv("../input/train.csv")
testDF = pd.read_csv('../input/test.csv')

# remove constant columns
colsToRemove = []
for col in trainDF.columns:
    if trainDF[col].std() == 0:
        colsToRemove.append(col)

trainDF.drop(colsToRemove, axis=1, inplace=True)
testDF.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns
colsToRemove = []
columns = trainDF.columns
for i in range(len(columns)-1):
    v = trainDF[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v,trainDF[columns[j]].values):
            colsToRemove.append(columns[j])
            
trainDF.drop(colsToRemove, axis=1, inplace=True)
testDF.drop(colsToRemove, axis=1, inplace=True)

trainLabels = trainDF.TARGET.values
test_ids = testDF.ID

trainDF = trainDF.drop(['ID','TARGET'], axis=1)
testDF = testDF.drop(['ID'], axis=1)

print (trainDF.shape)
print (testDF.shape)

slct = SelectFpr(alpha = 0.001) #Filter: Select the pvalues below alpha based on a FPR test.
trainFeatures = slct.fit_transform(trainDF, trainLabels)

print (trainFeatures.shape)

colsToRetain = slct.get_support(indices = True)
columns = trainDF.columns
colsToRemove = []
for i in range(len(columns)):
    if i not in colsToRetain:
        colsToRemove.append(columns[i])
testFeatures = testDF.drop(colsToRemove, axis=1).values
print (testFeatures.shape)


clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=550, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
X_train, X_test, y_train, y_test = train_test_split(trainFeatures, trainLabels, test_size=0.3)

# fitting
clf.fit(trainFeatures, trainLabels, eval_metric="auc", early_stopping_rounds=20, eval_set=[(X_test, y_test)])
test_pred = clf.predict_proba(testFeatures)[:,1]

submission = pd.DataFrame({"ID":test_ids, "TARGET":test_pred})
submission.to_csv("submission-fpr.csv", index=False)
