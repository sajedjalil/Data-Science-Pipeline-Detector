import numpy as np
import pandas as pd
import time

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

random_state = 13
np.random.seed(random_state)

print('read data')
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
test_ID = df_test['ID_code'].values
Y = df_train.target.values.astype(np.float32)
df_train = df_train.drop(['ID_code','target'], axis=1)
df_test = df_test.drop(['ID_code'], axis=1)
df_all = pd.concat([df_train,df_test], axis=0, sort=False, ignore_index=True)
del df_train, df_test
enginering_feats = [('var_26','var_44'),('var_44','var_123'),('var_44','var_155')]

print('start training of LightGBM...')
import lightgbm as lgb
from sklearn.model_selection import KFold

start_tiem = time.time()
n_predict = 0
valid = np.zeros( (len(test_ID),) )

def lgb_roc_score(y_hat, data):
    y_true = data.get_label()
    return 'roc', roc_auc_score(y_true, y_hat), True

for fe_id, fe in enumerate(enginering_feats):
    # Magic Feature Enginering
    df_e = df_all.copy()
    df_e['%s_plus_%s'%fe] = df_e[fe[0]]+df_e[fe[1]]
    df_e['%s_minus_%s'%fe] = df_e[fe[1]]-df_e[fe[0]]
    df_e = df_e.drop(list(fe),axis=1)
    Xp = df_e.values
    _X = Xp[:len(Y)]
    Xt = Xp[len(Y):]
    X = _X
    del df_e, _X
    
    for fold_id, (IDX_train, IDX_test) in enumerate(KFold(n_splits=6, random_state=fe_id+random_state, shuffle=True).split(Y)):
    	X_train = X[IDX_train]
    	X_test = X[IDX_test]
    	Y_train = Y[IDX_train]
    	Y_test = Y[IDX_test]
    
    	lgb_params = {
            "objective" : "binary",
            "metric" : "roc",
            "max_depth" : 2,
            "num_leaves" : 2,
    		"learning_rate" : 0.055,
    		"bagging_fraction" : 0.3,
    		"feature_fraction" : 0.15,
    		"lambda_l1" : 5,
    		"lambda_l2" : 5,
    		"bagging_seed" : fe_id+10*fold_id+random_state,
    		"verbosity" : 1,
    		"seed": fe_id+10*fold_id+random_state
    	}
    
    	lgtrain = lgb.Dataset(X_train, label=Y_train)
    	lgtest = lgb.Dataset(X_test, label=Y_test)
    	evals_result = {}
    	lgb_clf = lgb.train(lgb_params, lgtrain, 35000, 
    						valid_sets=[lgtrain, lgtest], 
    						early_stopping_rounds=500, 
    						verbose_eval=2000, 
    						feval=lgb_roc_score,
    						evals_result=evals_result)
    	valid += lgb_clf.predict( Xt ).reshape((-1,))
    	n_predict += 1
    	#if time.time() - start_tiem > 7180:
    	#	break

valid = np.clip( valid / n_predict, 0.0, 1.0 )
print('save result.')
pd.DataFrame({'ID_code':test_ID,'target':valid}).to_csv('submission.csv',index=False)
print('done.')

