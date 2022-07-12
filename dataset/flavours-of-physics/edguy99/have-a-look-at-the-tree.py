# Note: Kaggle only runs Python 3, not Python 2
import pandas
from sklearn.ensemble import RandomForestClassifier
import xgboost

print("Have a look at the tree")

training = pandas.read_csv("../input/training.csv")
test  = pandas.read_csv("../input/test.csv")
# print(training.describe(4))

print("Eliminate SPDhits, which makes the agreement check fail")
traininglist = list(training.columns[1:-5])
# 1:-5 got 0.979941, n_estimators=150
# 1:-5 got 0.980000, n_estimators=100
# 1:-5 got 0.979807, n_estimators=80
# 1:-5 got 0.??????, n_estimators=50 Evaluation Exception: The value 0.0038750675382139 is above the correlation threshold of 0.002..
# 1:-5 got 0.??????, n_estimators=10 Evaluation Exception: The value 0.0038750675382139 is above the correlation threshold of 0.002..
print(traininglist[0])
print(traininglist)

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=15, random_state=1)
rf.fit(training[traininglist], training["signal"])

print("Tree structure for Random Forest model")
from sklearn import tree
i_tree = 0
for tree_in_forest in rf.estimators_:
    print( str(i_tree)+" ========================= Tree node start ==========================" )
    print( "node_count:" + str(tree_in_forest.tree_.node_count) + ", max_depth:" + str(tree_in_forest.tree_.max_depth) )
    print( " children_left[0]:" + str(tree_in_forest.tree_.children_left[0]) + ", children_left[1]:" + str(tree_in_forest.tree_.children_left[1]) + ", children_left[2]:" + str(tree_in_forest.tree_.children_left[2]) )
    print( " children_right[0]:" + str(tree_in_forest.tree_.children_right[0]) + ", children_right[1]:" + str(tree_in_forest.tree_.children_right[1]) + ", children_right[2]:" + str(tree_in_forest.tree_.children_right[2]) )
    
    print( "threshold:" )
    print( tree_in_forest.tree_.threshold )

    i_tree = i_tree + 1


print("Make predictions on the test set")
test_probs = rf.predict_proba(test[traininglist])[:,1]
submission = pandas.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_submission.csv", index=False)

print("done...")


