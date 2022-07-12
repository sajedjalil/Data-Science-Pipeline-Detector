# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
DATA_DIR = "../input"
RESULT_DIR = "./"

TRAIN_DATA = "{0}/train.csv".format(DATA_DIR)
TEST_DATA = "{0}/test.csv".format(DATA_DIR)

MOD = os.path.basename(__file__).split(".")[0]
#TIME_STAMP = datetime.now().strftime('_%y%m%d_%H%M%S')
# RESULT_FILE = os.path.join(RESULT_DIR, MOD + TIME_STAMP + ".csv")
RESULT_FILE = os.path.join(RESULT_DIR, "submission.csv")

ID_COL = "id"
FEATURE_COLS = ["bone_length", "rotting_flesh", "hair_length", "has_soul"]
TARGET_COL = "type"

READ_COLS = [ID_COL, TARGET_COL] + FEATURE_COLS
train = pd.read_csv(TRAIN_DATA, usecols=READ_COLS)

READ_COLS = [ID_COL] + FEATURE_COLS
test = pd.read_csv(TEST_DATA, usecols=READ_COLS)

# Get target values...
y = train[TARGET_COL].ravel()
y_le = LabelEncoder()
y = y_le.fit_transform(y)

train = np.array(train[FEATURE_COLS])


RANDOM_STATE = 0
xgb_params = {
    'seed': RANDOM_STATE,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 2,
    'gamma': 1,
    'alpha': .1,
    'eta': 0.001,
}

num_rounds = 1000
dtrain = xgb.DMatrix(train, label=y)

dtest = xgb.DMatrix(np.array(test[FEATURE_COLS]))

res = xgb.cv(xgb_params, dtrain, num_boost_round=num_rounds, nfold=5, seed=RANDOM_STATE, stratified=True,
             verbose_eval=True, show_stdv=True)

cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('CV-Mean:\t%.5f +/- %.5f' % (cv_mean, cv_std))

watchlist = [(dtrain, 'train')]
model = xgb.train(xgb_params, dtrain, num_boost_round=num_rounds, evals=watchlist, verbose_eval=False)

y_pred = model.predict(dtest)

y_pred = y_le.inverse_transform(y_pred.astype(int))

submission = pd.DataFrame({
    "id": test["id"],
    "type": y_pred
})
submission.to_csv(RESULT_FILE, index=False)
