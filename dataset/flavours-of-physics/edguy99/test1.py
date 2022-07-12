# Note: Kaggle only runs Python 3, not Python 2
import pandas
from sklearn.ensemble import RandomForestClassifier
import xgboost

print("First look at data")

training = pandas.read_csv("../input/training.csv")
test  = pandas.read_csv("../input/test.csv")
print(training.describe(4))

print("Eliminate SPDhits, which makes the agreement check fail")
traininglist = list(training.columns[1:-4])
# 1:-5 got 0.983029 "max_depth": 6, "eta": 0.3, 1/4normal 3/4boosted
# 1:-5 got 0.982855 "max_depth": 6, "eta": 0.3, 1/3normal 2/3boosted 
# 1:-5 got 0.982410 "max_depth": 6, "eta": 0.3
# 1:-5 got 0.982273 "max_depth": 8, "eta": 0.2
# 1:-5 got 0.982253 "max_depth": 5 ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP', 'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree', 'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3', 'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT', 'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof', 'p0_IP', 'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 'p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta']
# 1:-5 got 0.981678 "max_depth": 8, "eta": 0.5
# 1:-5 got 0.980000
# 1:-30 got 0.971988
# 5:7 got 0.936119
# 5:13 got 0.962145 ['IP', 'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree', 'IP_p0p2']
# 5:20 got 0.970656 ['IP', 'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree', 'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf']
# 5:32 got 0.977144 ['IP', 'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree', 'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3', 'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT', 'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof', 'p0_IP']
print(traininglist[0])
print(traininglist)

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(training[traininglist], training["signal"])


print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 6,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgboost.train(params, xgboost.DMatrix(training[traininglist], training["signal"]), num_trees)

print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[traininglist])[:,1] +
              gbm.predict(xgboost.DMatrix(test[traininglist]))*3)/4

submission = pandas.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_submission.csv", index=False)

print("done...")


