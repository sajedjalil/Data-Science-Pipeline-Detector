# baseed on  https://www.kaggle.com/fayzur/lgb-bayesian-parameters-finding-rank-average

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



import time
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer

import xgboost as xgb
import lightgbm as lgb

import warnings

print('read data')
df_train = pd.read_csv('../input/train.csv')

test=False
df_train_sampled=df_train.sample(frac=0.1) # downsampling options for test purposes
if test==True: 
    df_train=df_train_sampled
    
df_test = pd.read_csv('../input/test.csv')
liste_column=df_train.columns[2:]
# Add some features
for df in [df_train, df_test]:
    df['the_median'] = df.median(axis=1)
    df['the_mean'] = df.mean(axis=1)
    df['the_sum'] = df.sum(axis=1)
    df['the_std'] = df.std(axis=1)
    df['the_kur'] = df.kurtosis(axis=1)

 
y_train = df_train.target
X_train = df_train.drop(['ID_code','target'], axis=1)

test_ID = df_test['ID_code'].values
df_test = df_test.drop(['ID_code'], axis=1)


# bayesian optimisation

print('bayesian optimisation ')

bayesian_tr_index, bayesian_val_index  = list(StratifiedKFold(n_splits= 5, shuffle=True, random_state=13).split(X_train, y_train))[0]
'''
def XGB_bayesian(
    n_estimators,  # int
    max_depth,  # int
    learning_rate,     
    subsample,
    colsample_bytree,
    gamma,
    reg_lambda,
    reg_alpha,
    scale_pos_weight
    ):
    
    # XGB expects next three parameters need to be integer. So we make them integer
    n_estimators = int( n_estimators)
    max_depth = int(max_depth)

    assert type(n_estimators) == int
    assert type(max_depth) == int

    param = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'verbose': True,
        'metric': 'auc',
        'tree_method': 'gpu_hist'

    }    

    xg_train = xgb.DMatrix(X_train.iloc[bayesian_tr_index].values,
                           label=y_train.iloc[bayesian_tr_index].values
                           )
    xg_valid = xgb.DMatrix(X_train.iloc[bayesian_val_index].values,
                           label=y_train.iloc[bayesian_val_index].values
                           )   

    num_round = 5000
    watchlist=[(xg_train,'train'),(xg_valid,'valid')]
    
    clf = xgb.train(param, xg_train, num_round, evals=watchlist, verbose_eval=1, early_stopping_rounds = 50)
    
    predictions = clf.predict(xg_valid)# , num_iteration=clf.best_iteration)   
    
    score = metrics.roc_auc_score(y_train.iloc[bayesian_val_index].values, predictions)
    
    return score

bounds_XGB = {
    'n_estimators': (200, 1200), 
    'max_depth': (1, 3),  
    'learning_rate': (0.01, 0.8),
    'subsample': (0.7,0.99),    
    'colsample_bytree': (0.001, 0.3),#si bcp de colonne, si peu de colonne alors 0.8 1.
    'gamma': (0, 0.1), 
    'reg_lambda': (0.5, 2), 
    'reg_alpha': (0.5, 2),
    'scale_pos_weight':(0.5,2), # ratio sommes positif sur sommes negatif
}
from bayes_opt import BayesianOptimization
XGB_BO = BayesianOptimization(XGB_bayesian, bounds_XGB, random_state=13)

init_points = 3  
n_iter = 5

print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    XGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
'''
#XGB_BO.max['params'] 

# best parameters,filled manually because of integers

'''
from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val=train_test_split(X_train,y_train,test_size=0.001,random_state=13)

print("definition classificateur")
param_xgb={'colsample_bytree': 0.2168834506255672,
 'gamma': 0.003503652410143732,
 'learning_rate': 0.24577508200245174,
 'max_depth': 1,
 'n_estimators': 1057,
 'reg_alpha': 1.0592810418122112,
 'reg_lambda': 1.5197719273671455,
 'scale_pos_weight': 0.8844199239899452,
 'subsample': 0.8007985523942226}
best_classificateur=xgb.XGBClassifier(**param_xgb)
print('fit XGB classifier ')
n_fold=5
kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=13)
cvscores = []
predictions=0
for train, test in kfold.split(X_train,y_train):
    best_classificateur.fit(X_train.iloc[train],y_train.iloc[train])
    predictions+=best_classificateur.predict_proba(df_test)/n_fold
    


sub_df=pd.DataFrame({"ID_code": test_ID})
sub_df["target"]=predictions[:,1]
sub_df.to_csv("submission_08_03_XGBoostalone.csv",index=False)
'''
# Naive Bayes
print('fit Naive Bayes ')
#predic_X_train_XGB=best_classificateur.predict_proba(X_train)
'''
pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
pipeline.fit(X_tr, y_tr)

pipeline.score(X_val,y_val)
predic_X_train_gaussian=pipeline.predict_proba(X_train)
predictions_gaussian=pipeline.predict_proba(df_test)
'''
'''
def LGB_bayesian(
    bagging_freq,  # int
    bagging_fraction,
    feature_fraction,     
    learning_rate,
    min_data_in_leaf, #int
    min_sum_hessian_in_leaf,
    num_leaves): # int
    
    # LGB expects next three parameters need to be integer. So we make them integer
    bagging_freq = int(bagging_freq)
    min_data_in_leaf=int(min_data_in_leaf)
    num_leaves=int(num_leaves)
    assert type(bagging_freq) == int
    assert type(min_data_in_leaf) == int
    assert type(num_leaves)==int
    param = {
        'bagging_freq': bagging_freq,
        'bagging_fraction':bagging_fraction,
        'boost_from_average' :'false',
        'boost':'gbdt',
        'feature_fraction':feature_fraction,
        'learning_rate':learning_rate,
        'max_depth':-1,
        'metric':'auc',
        'min_data_in_leaf':min_data_in_leaf,
        'min_sum_hessian_in_leaf':min_sum_hessian_in_leaf,
        'num_leaves':num_leaves,
        'tree_learner':'serial',
        'objective': 'binary',
        'num_threads': 8, 
        "device" : "cpu"

    }    

    lgb_train = lgb.Dataset(X_train.iloc[bayesian_tr_index].values,
                           label=y_train.iloc[bayesian_tr_index].values
                           )
    lgb_valid = lgb.Dataset(X_train.iloc[bayesian_val_index].values, label=y_train.iloc[bayesian_val_index].values
                           )   

    num_round = 5000
    clf = lgb.train(param, lgb_train, 1000000, valid_sets = [lgb_train, lgb_valid], verbose_eval=5, early_stopping_rounds = 4000)
    
    predictions = clf.predict(X_train.iloc[bayesian_val_index].values)# , num_iteration=clf.best_iteration)   
    
    score = metrics.roc_auc_score(y_train.iloc[bayesian_val_index].values, predictions)
    
    return score

Bounds_LGB = {
    'bagging_freq': (3,6),  'bagging_fraction': (0.2,0.4),
    'feature_fraction': (0.03,0.05), 'learning_rate':(0.007,0.009),
    'min_data_in_leaf': (70,90),     
    'min_sum_hessian_in_leaf': (9.,11),'num_leaves': (11,14)
}
 '''
#LGB_BO = BayesianOptimization(LGB_bayesian, Bounds_LGB, random_state=13)

init_points = 3  
n_iter = 5

print('-' * 130)
'''
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
LGB_BO.max['params']
'''
params_LGB={'bagging_freq': 6,
'learning_rate': 0.0089, 'bagging_fraction': 0.348,
'num_leaves': 13,'metric':'auc','boost_from_average':'false','boost': 'gbdt',
'feature_fraction': 0.0448,'max_depth': -1, 'min_data_in_leaf': 90,
'min_sum_hessian_in_leaf': 9, 'num_threads': 8,'tree_learner': 'serial','objective': 'binary','verbosity': -10}
print('fit LGB classifier ')
n_fold=5
oof=np.zeros(len(X_train))
np.zeros(len(df_test))
kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=13)

cvscores = []
predictions=0
for trn_idx, val_idx in kfold.split(X_train,y_train):
    trn_data=lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
    val_data=lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])
    clf = lgb.train(params_LGB, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions += clf.predict(df_test, num_iteration=clf.best_iteration) / n_fold
    
sub_df=pd.DataFrame({"ID_code": test_ID})
sub_df["target"]=predictions
sub_df.to_csv("submission_22_03_LGBStandtalone.csv",index=False)
'''
# Stacking
print('Stacking')
X_train_enriched=np.concatenate((predic_X_train_XGB[:,:1],predic_X_train_gaussian[:,:1]),axis=1)
df_test_enriched=np.concatenate((predictions[:,:1],predictions_gaussian[:,:1]),axis=1)
X_tr_enriched,X_val_enriched,y_tr_enriched,y_val_enriched=train_test_split(X_train_enriched,y_train,test_size=0.3,random_state=42)

# Rg 2 classifier : Neural Network
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers
import keras

print('NN Predictor')


n_fold=5
kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=13)
cvscores = []
predic_X_test_rg2=np.zeros((200000, 1))
for train, test in kfold.split(X_train_enriched,y_train):
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='tanh',kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid',kernel_initializer='glorot_normal'))
# patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# Compile model
    #sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
    model.fit(X_train_enriched[train], y_train[train], epochs=250, batch_size=1000, validation_data=(X_train_enriched[test], y_train[test]),callbacks=[es])
    scores = model.evaluate(X_train_enriched[test],  y_train[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    predic_X_test_rg2+=model.predict_proba(df_test_enriched)/n_fold
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

sub_df=pd.DataFrame({"ID_code": test_ID})
sub_df["target"]=predic_X_test_rg2
sub_df.to_csv("submission_XGB_Gaussian_NN_enriched_17_03.csv",index=False)
'''