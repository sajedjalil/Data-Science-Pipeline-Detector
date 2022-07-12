# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation
import operator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.linear_model import LogisticRegression

def normalize(training):
    training_norm = (training - training.min()) / (training.max() - training.min())
    return training_norm

# The competition datafiles are in the directory ../input
# List the files we have available to work with
print("> ls ../input")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read competition data files:
training = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
training = training.drop(["production","mass","min_ANNmuon"], axis=1)
training=normalize(training)

predictors=[]
for item in training:
    if item!='id' and item!='signal':
        predictors.append(item)

print(predictors)
#In this case we'll use a random forest, but this could be any classifier
alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=4, min_samples_leaf=3)
scores=cross_validation.cross_val_score(alg,training[predictors[0:-2]],training["signal"],cv=3)
print(scores.mean())

pva_preDict={}
for i in range(len(scores)):
    pva_preDict[scores[i]]=predictors[i]
print(pva_preDict)
newA = dict(sorted(pva_preDict.items(), key=operator.itemgetter(0),reverse=True)[:24])
print(newA)
newpredictors = []
for item in newA:
    if newA[item]!='signal':
        newpredictors.append(newA[item])
print(newpredictors)

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=100, max_depth=4),predictors],
    [LogisticRegression(random_state=1), newpredictors]
]
normtest=normalize(test)

full_predictions = []
for alg, newpredictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(training[newpredictors], training["signal"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(normtest[newpredictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pd.DataFrame({
        "id": test["id"],
        "signal": predictions
    })
submission.to_csv("kaggle.csv", index=False)

# Write summaries of the train and test sets to the log
print('\nSummary of train dataset:\n')
print(train.describe())
print('\nSummary of test dataset:\n')
print(test.describe())

