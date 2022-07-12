import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

#logger
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()

def read_data(nrows=None):
    logger.info('Input data')
    train_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv') #,nrows=nrows)
    synthetic = np.load('../input/list-of-fake-samples-and-public-private-lb-split/synthetic_samples_indexes.npy')
    train_df['isSynt'] = 0
    train_df['isSynt'][synthetic] = 1
    #train_df['oof'] = np.load('../input/santander-nn-embeddings/train_NN_embedding_oof.npy')
    test_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
    #test_df['oof'] = np.load('../input/santander-nn-embeddings/test_NN_embedding_pred.npy')
    return train_df, test_df
    
def process_data(train, test):
    from sklearn.preprocessing import StandardScaler
    traintest = pd.concat([train,test], axis=0, ignore_index=True).reset_index(drop=True)
    print(len(traintest))
    scaler = StandardScaler()
    cols = [c for c in train.columns if c not in ['ID_code', 'target']]
    traintest[cols] = scaler.fit_transform(traintest[cols])
    train = traintest[:len(train)].reset_index(drop=True) #360784
    test = traintest[len(train):].reset_index(drop=True)
    
    for i in range(0,200):
        varname = 'var_' + str(i)
        varname_bool = varname + '_bool'
        train[varname_bool] = np.where(train[varname]>=0, 1, 0)
        test[varname_bool] = np.where(test[varname]>=0, 1, 0)
    return train, test

def run_model(train_df, test_df):
    logger.info('Prepare the model')
    
    #Train only with bools...
    varname = ['ID_code', 'target', 'isSynt']
    #for i in range(0,200):
    #    varname.append('var_' + str(i))
    
    features = [c for c in train_df.columns if c not in varname]
    target = train_df['isSynt']
    logger.info('Run model')
    param = {
        'bagging_freq': 5, #5
        'bagging_fraction': 0.4, #0.4
        'boost_from_average':'true',
        'boost': 'gbdt',
        'feature_fraction': 0.05, # 0.045, 0.05 
        'learning_rate': 0.01, #0.0095
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 80, #80
        'min_sum_hessian_in_leaf': 10.0, #10.0
        'num_leaves': 21, #3 , 4
        'num_threads': 8,
        #'lambda_l2': 0.005,  #0.01 slow
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1
    }
    num_round = 8000
    folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
        print("Fold {}".format(fold_))
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000) #, early_stopping_rounds = 3500
        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
    print("CV score: {:<8.5f}".format(roc_auc_score(target, 1-oof)))
    np.save('test_LGBM_oof_fake.npy',oof)
    return predictions

def submit(test_df, predictions):
    logger.info('Prepare submission')
    #sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
    #sub["target"] = predictions
    #sub.to_csv("submission.csv", index=False)
    np.save('train_LGBM_pred_fake.npy',predictions)

def main(nrows=None):
    train_df, test_df = read_data(nrows)
    #train_df, test_df = process_data(train_df, test_df)
    predictions = run_model(train_df, test_df)
    submit(test_df, predictions)
    
if __name__ == "__main__":
    main()
    