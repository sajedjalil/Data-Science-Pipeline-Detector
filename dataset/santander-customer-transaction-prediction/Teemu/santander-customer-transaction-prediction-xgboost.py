############################################################################## Library
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import xgboost as xgb
############################################################################## Data
train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')
#Id column
train_csv.head()
train_id = train_csv['ID_code']
test_id = test_csv['ID_code']
#target column
target = train_csv['target']
#drop id-columns
variables =train_csv.drop(['ID_code', 'target'], axis=1, inplace = False)
variables.head()
############################################################################## Data Pre-processing
# Categorical data - check: none
variables.dtypes
# Empty values- check: none
variables.isnull().any()
#Filling empty values with mean:
#variables = variables.fillna(variables.mean())
#skewed features: none
#numeric_feats = variables.dtypes[variables.dtypes != "object"].index
#skewed_feats = variables[numeric_feats].apply(lambda x: skew(x.dropna()))
#skewed_feats = skewed_feats[abs(skewed_feats) > 0.55]
#skewed_feats 
#Normalizing skewed data:
#data_skewed = skewed_feats.index
#variables[data_skewed] = np.log1p(variables[data_skewed])
############################################################################## Model building
#X_train and y_train
X_train = variables
y_train = target
#X_test
variables_test = test_csv.drop('ID_code', axis=1, inplace = False)
X_test = variables_test
#XGboost
df_train = xgb.DMatrix(X_train, y_train)
df_test = xgb.DMatrix(X_test)

#crossvalidation
#Parameters for Tree Booster parameters https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster
params = {"eta":0.3, "gamma":10, "max_depth":5,  "min_child_weight":2, "subsample":0.5, 'objective': 'binary:hinge'}
model = xgb.cv(params, df_train,  num_boost_round=20, nfold=3, early_stopping_rounds=5, metrics={'error'})
#learning curve - modify: https://matplotlib.org/tutorials/introductory/pyplot.html
plt.plot(model['train-error-mean'], '-', model['test-error-mean'], '--')

#actual fitting:
model_xgb = xgb.XGBRegressor(max_depth=6, n_estimators=100, learning_rate=0.3, gamma=3, objective='binary:hinge') 
model_xgb.fit(X_train, y_train)
xgb_preds = model_xgb.predict(X_test)

############################################################################## Metrics
#Confusion Matrix:
Pred_train = pd.DataFrame(model_xgb.predict(variables))
Pred_train = np.select([Pred_train  <= .65, Pred_train >.65], [np.zeros_like(Pred_train ), np.ones_like(Pred_train )])
cm = confusion_matrix(target, Pred_train)
sn.heatmap(cm, annot=True)

#F1-score:
f1_score(target, Pred_train, average='binary')
##############################################################################

############################################################################## Results
solution = pd.DataFrame({"ID_code":test_id, "target":xgb_preds})
solution.to_csv("santander.csv", index = False)



