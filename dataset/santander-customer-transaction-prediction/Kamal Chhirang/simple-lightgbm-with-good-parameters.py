import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
#import xgboost as xgb
import catboost as cb
import pickle
import os
import gc
gc.enable()


def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    
    model = lgb.LGBMClassifier(objective = "binary", 
    boosting = "gbdt",
    metric="auc",
    boost_from_average=False,
    num_threads=8,
    learning_rate =0.0081,
    num_leaves =13,
    max_depth=-1,
    feature_fraction =0.041,
    bagging_freq =5,
    bagging_fraction =0.331,
    min_data_in_leaf =80,
    min_sum_hessian_in_leaf =10.0,
    verbosity =1,
    num_iterations =99999999,
    seed=44000)
    
     
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=3500, 
              early_stopping_rounds=3500)
                  
    cv_val = model.predict_proba(X_val)[:,1]
    
    #Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter+1)
    model.booster_.save_model(save_to)
    
    return cv_val
    
def train_stage(df_path, lgb_path):
    
    print('Load Train Data.')
    df = pd.read_csv(df_path)
    print('\nShape of Train Data: {}'.format(df.shape))
    
    y_df = np.array(df['target'])                        
    df_ids = np.array(df.index)                     
    df.drop(['ID_code', 'target'], axis=1, inplace=True)
    
    lgb_cv_result = np.zeros(df.shape[0])
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]
    
        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name='lgb')
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    auc_lgb  = round(roc_auc_score(y_df, lgb_cv_result),4)
    print('\nLightGBM VAL AUC: {}'.format(auc_lgb))
    return 0
    
    
def prediction_stage(df_path, lgb_path):
    
    print('Load Test Data.')
    df = pd.read_csv(df_path)
    print('\nShape of Test Data: {}'.format(df.shape))
    
    df.drop(['ID_code'], axis=1, inplace=True)
    
    lgb_models = sorted(os.listdir(lgb_path))
    lgb_result = np.zeros(df.shape[0])

    print('\nMake predictions...\n')
    
    for m_name in lgb_models:
        #Load LightGBM Model
        model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
        lgb_result += model.predict(df.values)

    lgb_result /= len(lgb_models)
    
    submission = pd.read_csv('../input/sample_submission.csv')
    submission['target'] = lgb_result
    submission.to_csv('lgb_starter_submission.csv', index=False)


    return 0
    
    
if __name__ == '__main__':
    
    train_path = '../input/train.csv'
    test_path  = '../input/test.csv'
    lgb_path = './lgb_models_stack/'

    #Create dir for models
    os.mkdir(lgb_path)

    print('Train Stage.\n')
    train_stage(train_path, lgb_path)
    
    print('Prediction Stage.\n')
    prediction_stage(test_path, lgb_path)
    
    print('\nDone.')