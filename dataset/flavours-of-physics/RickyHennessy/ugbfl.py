import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from hep_ml.uboost import uBoostClassifier
from hep_ml.gradientboosting import KnnFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier

print("Load the training/test data using pandas")
train = pd.read_csv("input/training.csv")
test  = pd.read_csv("input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
train_features = list(train.columns[1:-5])
uniform_features  = ["mass"]

n_estimators = 150
"""
print("Train uBoost Classifier")
base_tree = DecisionTreeClassifier(max_depth=4)
clf = uBoostClassifier(uniform_features, 
                       uniform_label=0,
                       n_estimators=n_estimators,
                       base_estimator=base_tree, 
                       train_features = train_features)
"""      

print("Train uGBFL")
flatnessloss = KnnFlatnessLossFunction(uniform_features, 
                                       fl_coefficient=3., 
                                       power=1.3, 
                                       uniform_label=1)
clf = UGradientBoostingClassifier(loss=flatnessloss, 
                                  max_depth=4, 
                                  n_estimators=n_estimators, 
                                  learning_rate=0.1, 
                                  train_features=train_features)
                 
clf.fit(train[train_features+uniform_features], train['signal'])

print("Make predictions on the test set")
test_probs = clf.predict_proba(test[train_features])[:,1]
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("uboost.csv", index=False)