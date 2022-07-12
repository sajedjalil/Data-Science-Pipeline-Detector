# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from numba import jit
import lightgbm as lgb

#print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv", na_values='-1')
test = pd.read_csv("../input/test.csv", na_values='-1')

@jit
def gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini    

def gini_normalized(a, p):
     return gini(a, p) / gini(a, a)
    
def gini_lgb(preds, dtrain):
    y = dtrain
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True    
    
def gini_lgb_train(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True     

# Preprocessing 
test_ids = test['id'].values
y_train  = train['target']
X_train  = train.drop(['target','id'], axis = 1)
X_test   = test.drop(['id'], axis = 1)

col_to_drop = X_train.columns[X_train.columns.str.startswith('ps_calc_')]
X_train     = X_train.drop(col_to_drop, axis=1)  
X_test      = X_test.drop(col_to_drop, axis=1)  

X_train = X_train.replace(-1, np.nan)
X_test  = X_test.replace(-1, np.nan)

cat_features = [col for col in X_train.columns if '_cat' in col]
for column in cat_features:
	temp = pd.get_dummies(pd.Series(X_train[column]))
	X_train = pd.concat([X_train,temp],axis=1)
	X_train = X_train.drop([column],axis=1)
    
for column in cat_features:
	temp = pd.get_dummies(pd.Series(X_test[column]))
	X_test = pd.concat([X_test,temp],axis=1)
	X_test = X_test.drop([column],axis=1)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
print('Training data: ', X_train.values.shape, ',  Test data: ', X_test.values.shape)

# Training
n_splits = 3
splitter = model_selection.StratifiedShuffleSplit(n_splits=n_splits)
scores = []

submission = pd.DataFrame({
    'id': test_ids,
    'target': 0
})

params1 = {'learning_rate': 0.09, 'max_depth': 6, 'boosting': 'gbdt', 'objective': 'binary', 
          'metric': 'auc', 'num_leaves': 40, 'min_data_in_leaf': 200, 'max_bin': 100, 'colsample_bytree' : 0.5,   
          'subsample': 0.7, 'subsample_freq': 2, 'verbose':-1, 'is_training_metric': False, 'seed': 1974}
params2 = {'learning_rate': 0.12, 'max_depth': 4, 'verbose':-1, 'num_leaves':16,
           'is_training_metric': False, 'seed': 1974} 
params3 = {'learning_rate': 0.11, 'subsample': 0.8, 'boosting': 'gbdt', 'objective': 'binary', 
          'metric': 'auc', 'subsample_freq': 10, 'colsample_bytree': 0.6, 'max_bin': 10, 
           'min_child_samples': 500,'verbose':-1, 'is_training_metric': False, 'seed': 1974}  

num_models = 3
log_model       = LogisticRegression()
X_logreg_train  = np.zeros((X_train.shape[0], n_splits * num_models))
X_logreg_test   = np.zeros((X_test.shape[0], n_splits * num_models))

lgb_params1 = {}
lgb_params1['learning_rate'] = 0.02
lgb_params1['n_estimators'] = 300
lgb_params1['max_bin'] = 10
lgb_params1['subsample'] = 0.7
lgb_params1['subsample_freq'] = 12
lgb_params1['colsample_bytree'] = 0.7   
lgb_params1['min_child_samples'] = 600
lgb_params1['seed'] = 1974


lgb_params2 = {}
lgb_params2['n_estimators'] = 1500
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 1974


lgb_params3 = {}
lgb_params3['n_estimators'] = 1500
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 1974

model1 = lgb.LGBMClassifier(**lgb_params1)
model2 = lgb.LGBMClassifier(**lgb_params2)
model3 = lgb.LGBMClassifier(**lgb_params3)

for i, (fit_index, val_index) in enumerate(splitter.split(X_train, y_train)):
    X_fit = X_train.iloc[fit_index,:].copy()
    y_fit = y_train.iloc[fit_index].copy()
    X_val = X_train.iloc[val_index,:].copy()
    y_val = y_train.iloc[val_index].copy()

    model1.fit(
        X_fit,
        y_fit)
 #       eval_set=[(X_val, y_val)],
 #       eval_metric=gini_lgb,
 #       early_stopping_rounds=50,
 #       verbose=False  )
 #   model1 = lgb.train(params1, 
 #                 train_set       = lgb.Dataset(X_fit, label=y_fit), 
 #                 num_boost_round = 200,
 #                 valid_sets      = lgb.Dataset(X_val, label=y_val),
 #                 verbose_eval    = 50, 
 #                 feval           = gini_lgb,
 #                 early_stopping_rounds = 50)
    #y_val_predprob1 = model1.predict(X_val, num_iteration=model1.best_iteration)
    y_val_predprob1 = model1.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob1)
    scores.append(score)
    print('Fold {} model 1: {} gini'.format(i+1, score))
    #x_test_pred1   = model1.predict(X_test, num_iteration=model1.best_iteration)
    x_test_pred1   = model1.predict_proba(X_test)[:,1] 
    #x_train_pred1  = model1.predict(X_train, num_iteration=model1.best_iteration)
    x_train_pred1  = model1.predict_proba(X_train)[:,1] 
    X_logreg_test[:, i * num_models]  = x_test_pred1
    X_logreg_train[:, i * num_models] = x_train_pred1
    
    model2.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric=gini_lgb,
        early_stopping_rounds=50,
        verbose=False  )
    y_val_predprob2 = model2.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob2)
    scores.append(score)
    print('Fold {} model 2: {} gini'.format(i+1, score))
    x_test_pred2   = model2.predict_proba(X_test)[:,1] 
    x_train_pred2  = model2.predict_proba(X_train)[:,1] 
    X_logreg_test[:, i * num_models + 1]  = x_test_pred2
    X_logreg_train[:, i * num_models + 1] = x_train_pred2
    
    model3.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric=gini_lgb,
        early_stopping_rounds=50,
        verbose=False  )
    y_val_predprob3 = model3.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob3)
    scores.append(score)
    print('Fold {} model 3: {} gini'.format(i+1, score))
    x_test_pred3   = model3.predict_proba(X_test)[:,1] 
    x_train_pred3  = model3.predict_proba(X_train)[:,1] 
    X_logreg_test[:, i * num_models + 2]  = x_test_pred3
    X_logreg_train[:, i * num_models + 2] = x_train_pred3

log_model.fit(X = X_logreg_train, y = y_train)
submission['target'] = log_model.predict_proba(X_logreg_test)[:,1]
    
# Submission:
submission.to_csv('lightgbm.csv', index=False)