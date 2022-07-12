# Will Koehrsen's amazing notebook on 'Automatic Feature Engineering Basics' generates 1000+ deeply synthesized features
# But, this feature space has a huge dimensionality so how can we effectively train a classifier on it?
# LightGBM can prevent overfitting thorugh extensive sets of regularization options
# But, even then, there are a large number of hyperparameters that can't be fully evaluated through GridSearchCV()
# Bayesian Optimization comes to the rescue!
# BayesOpt first calculates the response for a set of random points
# Then, through the magic of Bayes Thereom, it continuously updates the prior/posteriar distributions searching for the optimal set of parameters

# Important Notes:  
# To operate in Kaggle Kernels quickly, only the first 1000 rows are loaded up right now. 
# You'll need to raise that number substantially, preferably letting it run overnight on a more powerful machine
# For BayesOpt, experiment with the number of random seed points and random iterations
# Uncomment the last block of code to generate predictions


#####################################################################################################################

import pandas as pd;
import numpy as np;
import seaborn as sns;
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
import os
print(os.listdir("../input"))

#####################################################################################################################

print('Importing Training Data')

train = pd.read_csv('../input/home-credit-default-risk-feature-tools/feature_matrix.csv', nrows=10000)
#train = pd.read_csv('../input/home-credit-default-risk-feature-tools/feature_matrix.csv')

sub = pd.read_csv('../input/home-credit-default-risk/sample_submission.csv')

test  = train[train['set'] == 'test']
train = train[train['set'] == 'train']

y = train['TARGET']

train = train.drop(columns = ['set', 'TARGET', 'SK_ID_CURR'])

#assert (test['SK_ID_CURR'].values == sub['SK_ID_CURR'].values).all()

#####################################################################################################################

print('Converting Categorical Variables')

train.loc[train['CODE_GENDER'] =='F', 'CODE_GENDER'] = 0;
train.loc[train['CODE_GENDER'] =='M', 'CODE_GENDER'] = 1;

train.loc[train['FLAG_OWN_CAR'] =='N', 'FLAG_OWN_CAR'] = 0;
train.loc[train['FLAG_OWN_CAR'] =='Y', 'FLAG_OWN_CAR'] = 1;

train.loc[train['FLAG_OWN_REALTY'] =='N', 'FLAG_OWN_REALTY'] = 0;
train.loc[train['FLAG_OWN_REALTY'] =='Y', 'FLAG_OWN_REALTY'] = 1;

test.loc[test['CODE_GENDER'] =='F', 'CODE_GENDER'] = 0;
test.loc[test['CODE_GENDER'] =='M', 'CODE_GENDER'] = 1;

test.loc[test['FLAG_OWN_CAR'] =='N', 'FLAG_OWN_CAR'] = 0;
test.loc[test['FLAG_OWN_CAR'] =='Y', 'FLAG_OWN_CAR'] = 1;

test.loc[test['FLAG_OWN_REALTY'] =='N', 'FLAG_OWN_REALTY'] = 0;
test.loc[test['FLAG_OWN_REALTY'] =='Y', 'FLAG_OWN_REALTY'] = 1;

cat_columns = train.select_dtypes(['object']).columns

train[cat_columns] = train[cat_columns].astype('category')
train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)

test[cat_columns] = test[cat_columns].astype('category')
test[cat_columns] = test[cat_columns].apply(lambda x: x.cat.codes)

train[cat_columns] = train[cat_columns].astype('category')
test[cat_columns]  = test[cat_columns].astype('category')

test = test.drop(columns = ['set', 'TARGET', 'SK_ID_CURR'])

#assert (train.columns.values == test.columns.values).all()

#####################################################################################################################

print('Filling in Missing Values in Training Data')

train_nan = train.isnull().sum(axis=0);
columns_to_fill_in = train_nan[train_nan != 0].index.values;

for column in tqdm(columns_to_fill_in):
    if train[column].dtype == 'O':
        train[column] = train[column].fillna(stats.mode(train[column].dropna().values)[0][0]);  
    else:
        train[column] = train[column].fillna(np.median(train[column].dropna()));
        
#####################################################################################################################

print('Filling in Missing Values in Testing Data')

test_nan = test.isnull().sum(axis=0);
columns_to_fill_in = test_nan[test_nan != 0].index.values;

for column in tqdm(columns_to_fill_in):
    if test[column].dtype == 'O':
        test[column] = test[column].fillna(stats.mode(test[column].dropna().values)[0][0]);  
    else:
        test[column] = test[column].fillna(np.median(test[column].dropna()));
        
#####################################################################################################################

def lgb_evaluate(
                 learning_rate,
                 num_leaves,
                 min_split_gain,
                 max_depth,
                 subsample,
                 subsample_freq,
                 lambda_l1,
                 lambda_l2,
                 feature_fraction,
                ):

    clf = lgb.LGBMClassifier(num_leaves              = int(num_leaves),
                             max_depth               = int(max_depth),
                             learning_rate           = 10**learning_rate,
                             n_estimators            = 500,
                             min_split_gain          = min_split_gain,
                             subsample               = subsample,
                             colsample_bytree        = feature_fraction,
                             reg_alpha               = 10**lambda_l1,
                             reg_lambda              = 10**lambda_l2,
                             subsample_freq          = int(subsample_freq),
                             verbose                 = -1
                            )
    
    scores = cross_val_score(clf, train, y, cv=5, scoring='roc_auc')

    return np.mean(scores)
    
#####################################################################################################################

lgbBO = BayesianOptimization(lgb_evaluate, {
                                            'learning_rate':           (-2, 0),
                                            'num_leaves':              (5, 50),
                                            'min_split_gain':          (0, 1),
                                            'max_depth':               (5, 30),
                                            'subsample':               (0.1, 1),
                                            'subsample_freq':          (0, 100),
                                            'lambda_l1':               (-2, 2),
                                            'lambda_l2':               (-2, 2),
                                            'feature_fraction':        (0.1, 1)
                                            })

#####################################################################################################################

print('Optimizing.......')

lgbBO.maximize(init_points=5, n_iter=5)

print(lgbBO.res['max'])

#####################################################################################################################

'''
print('Training on Full Data Set')

params = lgbBO.res['max']['max_params']

clf = lgb.LGBMClassifier(num_leaves              = params['num_leaves'],
                         max_depth               = params['max_depth'],
                         learning_rate           = 10**params['learning_rate'],
                         n_estimators            = 500,
                         min_split_gain          = params['min_split_gain'],
                         subsample               = params['subsample'],
                         colsample_bytree        = params['feature_fraction'],
                         lambda_l1               = 10**params['lambda_l1'],
                         lambda_l2               = 10**params['lambda_l2'],
                         subsample_freq          = params['subsample_freq']
                        )

clf.fit(train, y, eval_metric='auc', verbose=1)

pred = clf.predict_proba(test)

sub['TARGET'] = [p[1] for p in pred]

sub.to_csv('sub_BayesOpt.csv', index=False)
'''