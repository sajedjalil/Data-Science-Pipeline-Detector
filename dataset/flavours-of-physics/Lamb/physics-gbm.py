import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.cross_validation import  train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

print("Load the training/test data using pandas")
fullTrain = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
sp = np.random.rand(len(fullTrain)) < .8



print("Eliminate SPDhits, which makes the agreement check fail")
features = list(fullTrain.columns[1:-5])

train = fullTrain[sp]
val = fullTrain[~sp]

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Use validation set to find best weighting for RF/GBM combination")

def get_preds(toScore, rf, gbm, features, alpha):
    rfPred = alpha * rf.predict_proba(toScore[features])[:,1]
    gbmPred = (1 - alpha) * gbm.predict(xgb.DMatrix(toScore[features]))
    return rfPred + gbmPred

def score_preds(true, pred):
    """Computes weighted AUC for scoring"""
    fpr, tpr, thresh = roc_curve(true, pred)

    regionA_x = [fpr[i] for i in range(len(tpr)) if tpr[i] < .2]
    regionA_y = [tpr[i] for i in range(len(tpr)) if tpr[i] < .2]
    if regionA_x == []:
        regionA_x = [0.0]
        regionA_y = [0.0]
    regionA_x.append(1.0)
    regionA_y.append(regionA_y[-1])
    regionA = np.trapz(regionA_y, regionA_x)

    regionB_x = [fpr[i] for i in range(len(tpr)) if tpr[i] >= .2 and tpr[i] < .4]
    regionB_y = [tpr[i] for i in range(len(tpr)) if tpr[i] >= .2 and tpr[i] < .4]
    regionB = np.trapz(regionB_y, regionB_x) - regionA
    if regionB_x == []:
        regionB_x = [0.0]
        regionB_y = [0.0]
    regionB_x.append(1.0)
    regionB_y.append(regionB_y[-1])

    regionC_x = [fpr[i] for i in range(len(tpr)) if tpr[i] >= .4 and tpr[i] < .6]
    regionC_y = [tpr[i] for i in range(len(tpr)) if tpr[i] >= .4 and tpr[i] < .6]
    if regionC_x == []:
        regionC_x = [0.0]
        regionC_y = [0.0]
    regionC_x.append(1.0)
    regionC_y.append(regionC_y[-1])
    regionC = np.trapz(regionC_y, regionC_x) -  regionA - regionB

    regionD_x = [fpr[i] for i in range(len(tpr)) if tpr[i] >= .6 and tpr[i] < .8]
    regionD_y = [tpr[i] for i in range(len(tpr)) if tpr[i] >= .6 and tpr[i] < .8]
    if regionD_x == []:
        regionD_x = [0.0]
        regionD_y = [0.0]
    regionD_x.append(1.0)
    regionD_y.append(regionD_y[-1])
    regionD = np.trapz(regionD_y, regionD_x) -  regionA -  regionB - regionC

    score = 2.0 * regionA + 1.5 * regionB + regionC + .5 * regionD
    return score

def find_alpha(val, rf, gbm, features):
    """Finds best weighting for predictions giving validation set"""
    true = val["signal"]
    bestScore = 0
    bestAlpha = None
    for alpha in np.linspace(0,1,101):
        print(alpha)
        pred = get_preds(val, rf, gbm, features, alpha)
        score = score_preds(true, pred)
        if score > bestScore:
            bestScore = score
            bestAlpha = alpha
    return bestAlpha

bestAlpha = find_alpha(val, rf, gbm, features) + .16 # Temp hack to simulate K-S correction

print("Retrain models on full training set")

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(fullTrain[features], fullTrain["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgb.train(params, xgb.DMatrix(fullTrain[features], fullTrain["signal"]), num_trees)

print("Make predictions on the test set")
test_probs = get_preds(test, rf, gbm, features, bestAlpha)
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)
