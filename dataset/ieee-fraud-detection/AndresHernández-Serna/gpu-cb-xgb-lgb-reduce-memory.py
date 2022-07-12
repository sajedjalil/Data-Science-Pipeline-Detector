#Kernel based: https://www.kaggle.com/bogorodvo/starter-code-saving-and-loading-lgb-xgb-cb !!!!
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pickle
import os
import gc
gc.enable()

#Reduce_memory
def reduce_memory(df):
    print("Reduce_memory...");
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    return df

#MODELS
#LightGBM Model
def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    model = lgb.LGBMClassifier(n_estimators=10000,learning_rate=0.02,max_depth=69, boosting_type='gbdt', objective= 'binary', metric='auc', seed= 4, num_leaves= 7, n_jobs=-1)
    model.fit(X_fit, y_fit,eval_set=[(X_val, y_val)],verbose=0,early_stopping_rounds=1000)
    cv_val = model.predict_proba(X_val)[:,1]
    #Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter+1)
    model.booster_.save_model(save_to)
    del X_fit, y_fit, X_val, y_val
    return cv_val
#XGBoost Model
def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    model = xgb.XGBClassifier(n_estimators=10000, max_depth=69, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, tree_method='gpu_hist')
    model.fit(X_fit, y_fit,eval_set=[(X_val, y_val)],verbose=0,early_stopping_rounds=1000)
    cv_val = model.predict_proba(X_val)[:,1]
    #Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter+1)
    pickle.dump(model, open(save_to, "wb"))
    del X_fit, y_fit, X_val, y_val
    return cv_val
#Catboost Model
def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    model = cb.CatBoostClassifier(iterations=10000,task_type = 'GPU')
    model.fit(X_fit, y_fit,eval_set=[(X_val, y_val)],verbose=0, early_stopping_rounds=1000)
    cv_val = model.predict_proba(X_val)[:,1]
    #Save Catboost Model          
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter+1)
    model.save_model(save_to, format="coreml",export_parameters={'prediction_type': 'probability'})
    del X_fit, y_fit, X_val, y_val
    return cv_val
#train
def train_stage(df_path,df2_path, lgb_path, xgb_path, cb_path):
    print('Load Train Data...')
    df = pd.read_csv(df_path,nrows=600000,index_col='TransactionID')
    df2 = pd.read_csv(df2_path,nrows=150000,index_col='TransactionID')
    
    print ("Merge test...");
    df = df.merge(df2, how='left', left_index=True, right_index=True)
    del df2
    print('\nShape of Train Data: {}'.format(df.shape))
    
    y_df = np.array(df['isFraud'])                        
    df_ids = np.array(df.index)                     
    df.drop(['TransactionDT', 'isFraud'], axis=1, inplace=True)
    df = df.fillna(-999)
    for f in df.columns:
        if df[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))
        
    df= reduce_memory(df)
    
    lgb_cv_result = np.zeros(df.shape[0])
    xgb_cv_result = np.zeros(df.shape[0])
    cb_cv_result  = np.zeros(df.shape[0])
    
    NumFold=4;
    skf = StratifiedKFold(n_splits=NumFold, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]
    
        print('LigthGBM')
        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name='lgb')
        print('CatBoost')
        cb_cv_result[ids[1]]  += fit_cb(X_fit,  y_fit, X_val, y_val, counter, cb_path,  name='cb')
        print('XGBoost')
        xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    auc_lgb  = round(roc_auc_score(y_df, lgb_cv_result),4)
    auc_xgb  = round(roc_auc_score(y_df, xgb_cv_result),4)
    auc_cb   = round(roc_auc_score(y_df, cb_cv_result), 4)
    
    auc_mean_lgb_cb = round(roc_auc_score(y_df, (lgb_cv_result+cb_cv_result)/2), 4)
    auc_mean_xgb_cb = round(roc_auc_score(y_df, (xgb_cv_result+cb_cv_result)/2), 4)
    auc_mean_xgb_lgb = round(roc_auc_score(y_df, (xgb_cv_result+lgb_cv_result)/2), 4)
    
    auc_mean = round(roc_auc_score(y_df, (lgb_cv_result+xgb_cv_result+cb_cv_result)/3), 4)
    
    print('\nLightGBM VAL AUC: {}'.format(auc_lgb))
    print('\nXGBoost  VAL AUC: {}'.format(auc_xgb))
    print('\nCatboost VAL AUC: {}'.format(auc_cb))
    print('\nMean Catboost+LightGBM VAL AUC: {}'.format(auc_mean_lgb_cb))
    print('\nMean XGBoost+Catboost, VAL AUC: {}\n'.format(auc_mean_xgb_cb))
    print('\nMean XGBoost+LightGBM, VAL AUC: {}\n'.format(auc_mean_xgb_lgb))
    print('\nMean XGBoost+Catboost+LightGBM, VAL AUC: {}\n'.format(auc_mean))
    
    del df
    return 0
#Prediction   
def prediction_stage(df_path,df2_path, lgb_path, xgb_path, cb_path):
    print('Load Test Data.')
    df = pd.read_csv(df_path,nrows=600000,index_col='TransactionID')
    df2 = pd.read_csv(df2_path,nrows=150000,index_col='TransactionID')
    
    print ("merge test");
    df = df.merge(df2, how='left', left_index=True, right_index=True)
    del df2
    print('\nShape of Test Data: {}'.format(df.shape))
    
    df.drop(['TransactionDT'], axis=1, inplace=True)
    df = df.fillna(-999)
    
    for f in df.columns:
        if df[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))
    del lbl
    df= reduce_memory(df)
    
    lgb_models = sorted(os.listdir(lgb_path))
    xgb_models = sorted(os.listdir(xgb_path))
    cb_models  = sorted(os.listdir(cb_path))
    
    lgb_result = np.zeros(df.shape[0])
    xgb_result = np.zeros(df.shape[0])
    cb_result  = np.zeros(df.shape[0])
    
    print('\nMake predictions...\n')
    
    print('With LightGBM...')
    for m_name in lgb_models:
        #Load LightGBM Model
        model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
        lgb_result += model.predict(df.values)
    del model
    print('With XGBoost...')    
    for m_name in xgb_models:
        #Load Catboost Model
        model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
        xgb_result += model.predict_proba(df.values)[:,1]
    del model
    print('With CatBoost...')        
    for m_name in cb_models:
        #Load Catboost Model
        model = cb.CatBoostClassifier()
        model = model.load_model('{}{}'.format(cb_path, m_name), format = 'coreml')
        cb_result += model.predict(df.values, prediction_type='Probability')[:,1]
    
    del df, model
    lgb_result /= len(lgb_models)
    xgb_result /= len(xgb_models)
    cb_result  /= len(cb_models)
    submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
    
    submission['isFraud'] = xgb_result
    submission.to_csv('xgb_starter_submission.csv')
    submission['isFraud'] = lgb_result
    submission.to_csv('lgb_starter_submission.csv')
    submission['isFraud'] = cb_result
    submission.to_csv('cb_starter_submission.csv')
    
    submission['isFraud'] = (xgb_result+cb_result)/2
    submission.to_csv('xgb_cb_starter_submission.csv')
    
    submission['isFraud'] = (lgb_result+xgb_result)/2
    submission.to_csv('xgb_lgb_starter_submission.csv')
    
    submission['isFraud'] = (lgb_result+cb_result)/2
    submission.to_csv('cb_lgb_starter_submission.csv')
    
    submission['isFraud'] = (lgb_result+xgb_result+cb_result)/3
    submission.to_csv('xgb_lgb_cb_starter_submission.csv')
    
    del lgb_result,xgb_result,cb_result
    
    return 0

if __name__ == '__main__':
    train_path = '../input/train_transaction.csv'
    train_identity='../input/train_identity.csv'
    
    test_path  = '../input/test_transaction.csv'
    test_identity='../input/test_identity.csv'
    
    lgb_path = './lgb_models_stack/'
    xgb_path = './xgb_models_stack/'
    cb_path  = './cb_models_stack/'
    #Create dir for models
    os.mkdir(lgb_path)
    os.mkdir(xgb_path)
    os.mkdir(cb_path)
    
    print('Train Stage.\n')
    train_stage(train_path,train_identity, lgb_path, xgb_path, cb_path)
    
    print('Prediction Stage.\n')
    prediction_stage(test_path,test_identity, lgb_path, xgb_path, cb_path)
    
    print('\nDone.')