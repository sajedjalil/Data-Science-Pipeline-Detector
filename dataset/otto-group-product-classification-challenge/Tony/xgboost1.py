import pandas as pd
import os
import xgboost as xgb

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

class_range = range(1, 10)
class_dict = {}
for n in class_range:
    class_dict['Class_{}'.format(n)] = n-1
train['target'] = train['target'].map(class_dict)

X_train = train.drop(["id", "target"], axis=1)
Y_train = train["target"].copy()
X_test = test.drop("id", axis = 1).copy()

params = {"objective": "multi:softprob", "eval_metric":"mlogloss", "num_class": 9}
T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)
gbm = xgb.train(params, T_train_xgb, 50)
Y_pred = gbm.predict(X_test_xgb)

submission = pd.DataFrame({ "id": test["id"]})

i = 0
for num in class_range:
    col_name = str("Class_{}".format(num))
    submission[col_name] = Y_pred[:,i]
    i = i + 1
    
submission.to_csv('prediction.csv', index=False)