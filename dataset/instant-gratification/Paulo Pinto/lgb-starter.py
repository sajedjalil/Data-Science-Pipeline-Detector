import pandas as pd

import numpy as np
np.random.random(42)

import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb

train = pd.read_csv("../input/train.csv") 
test = pd.read_csv("../input/test.csv")

submit = test[['id']].copy()
submit['target'] = 0

y = train["target"].values

train = train.drop(['id','target'], axis=1)
test = test.drop('id', axis=1)

for m in range(512):
    print('magic = ',m)
    idx_tr = (train['wheezy-copper-turtle-magic']==m)
    idx_te = (test['wheezy-copper-turtle-magic']==m)
    
    lgbtrain = lgb.Dataset(train[idx_tr].values, label=y[idx_tr])
    
    params = {
        'objective': 'binary', 
        'boost_from_average':'true',
        'boost': 'gbdt',
        'tree_learner': 'serial',
        'num_threads': 8,
        'max_depth': 7,
        'num_leaves': 63,
        'metric':'auc',
        'learning_rate': 0.05,
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 1, 
        'verbosity': 1
    }
    
    hist = lgb.cv(params,
                lgbtrain,
                num_boost_round=5000,
                nfold=5,
                stratified=True,
                shuffle=True,
                metrics=['auc'],
                fobj=None, 
                feature_name='auto', 
                categorical_feature='auto', 
                early_stopping_rounds=100, 
                verbose_eval=50, 
                show_stdv=False, 
                seed=42)
    
    best_iter = np.array(hist['auc-mean']).argmax()
    gbmdl = lgb.train(params, lgbtrain, num_boost_round=best_iter)
    #gbmdl.save_model('model_lgb{}.txt'.format(m))
    submit.loc[idx_te,'target'] = gbmdl.predict(test[idx_te].values,num_iteration=gbmdl.best_iteration)

submit.to_csv("submission.csv", float_format='%.4g', index=False)