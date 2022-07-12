import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import  datetime


def get_data():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    features = train.select_dtypes(include=['float']).columns
    features = np.setdiff1d(features,['ID','target'])

    test_ids = test.ID
    y_train = train.target

    x_train = train[features]
    x_test = test[features]

    return x_train, y_train, x_test, test_ids

ts = datetime.now()
x_train, y_train, x_test, test_ids = get_data()


xgb_params = {"objective": "binary:logistic", "max_depth": 10, "silent": 1}
num_rounds = 200

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)
gbdt = xgb.train(xgb_params, dtrain, num_rounds)

preds = gbdt.predict(dtest)


submission = pd.DataFrame({"ID": test_ids, "target": preds})
submission = submission.set_index('ID')
submission.to_csv('xgb_benchmark.csv')

te = datetime.now()
print('elapsed time: {0}'.format(te-ts))