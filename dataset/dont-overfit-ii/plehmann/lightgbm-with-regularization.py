import pandas as pd 
import numpy as np                                                                                   
import lightgbm as lgb


train = pd.read_csv('../input/train.csv')                                                            
train.pop('id')                                                                                      
target = train.pop('target').astype(int)                                                             
                                                                                                     
test = pd.read_csv('../input/test.csv')                                                              
ids = test.pop('id')                                                                                 
                                                                                                     

print('train',train.shape)
print('test',test.shape)
print(target.shape)
print(target.head())

lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        
        'learning_rate': 0.005,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 231,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 40,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.6,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
    'is_unbalanced':True
      
    }

train_data = lgb.Dataset(train, label=target.values)

            
model = lgb.train(lgb_params,
                    train_data,
                    num_boost_round=20000,
                    valid_sets = [train_data],
                    verbose_eval=500,
                    early_stopping_rounds = 500,
                    )
print('finished training')
y_pred = model.predict(test)
#y_pred = np.round(y_pred)
df = pd.DataFrame({'id': ids, 'target': y_pred}) 
df[['id', 'target']].to_csv('submission.csv', index=False)  
print('submitted file')