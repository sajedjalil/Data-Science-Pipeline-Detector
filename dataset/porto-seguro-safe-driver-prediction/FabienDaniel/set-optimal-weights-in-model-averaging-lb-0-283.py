import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
#=========================================================
# I. Load the data
#=========================================================
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#_________________________
def convert_dataframe(df):   
    col = [c for c in df.columns if not c.startswith('ps_calc_')]
    df_transformed = df[col].copy(deep = True)
    list_cols = []
    for col in df.columns:
        if not col.endswith('_cat'): continue
        test = pd.get_dummies(df[col])        
        columns = ['{}_{}'.format(col, s) for s in test.columns]
        test.columns = columns
        df_transformed = pd.concat([df_transformed, test], axis=1)
        list_cols.append(col)
    df_transformed.drop(list_cols, axis = 1, inplace = True)
    return df_transformed
#___________________________________________
df_transformed = convert_dataframe(df_train)
df_train = df_transformed.copy(deep = True)
df_transformed = convert_dataframe(df_test)
df_test = df_transformed.copy(deep = True)
#___________________________________________
y_train  = df_train['target'].values
id_train = df_train['id'].values
id_test  = df_test['id'].values
X_train = df_train.drop(['target', 'id'], axis=1)
X_test  = df_test.drop(['id'], axis=1)
del df_test, df_train
#=========================================================
# II. Define metrics functions
#=========================================================
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)
#__________________________ 
def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True
#__________________________ 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)
#=========================================================
# III. Run LGBM models
#=========================================================
params = {'learning_rate': 0.1,
          'subsample': 0.8,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 20.0,
          'reg_lambda': 0,
          'max_depth' : 4,
          'num_leaves': 16,        
          'min_data_in_leaf': 150, 
          'boosting': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'is_training_metric': False,
          'seed': 99,
          'verbose': -1}
#____________
NB_models  = 2
model_seed = [200, 211]
path_model = ['MOD_1', 'MOD_2']
#____________________________
start_time = time.time()
kfold = 5
MAX_TREES = 350
#________________________________________________
for num_mod, model_name in enumerate(path_model):    
    print(50*'_'+'\nModel nº: {} -> {}'.format(num_mod, model_name))
    sss = StratifiedKFold(n_splits = kfold, shuffle = True, random_state=model_seed[num_mod])
    #___________________________________________________________________________
    for j, (train_index, valid_index) in enumerate(sss.split(X_train, y_train)):    
        print('\n[Fold {}/{}]'.format(j + 1, kfold))
        #_________________________________________________________________
        x_tr, x_valid = X_train.loc[train_index], X_train.loc[valid_index]
        y_tr, y_valid = y_train[train_index], y_train[valid_index]
        #________________________________________________________________________________________________        
        mdl = lgb.train(params, lgb.Dataset(x_tr, label=y_tr), MAX_TREES, lgb.Dataset(x_valid, label=y_valid),
                         verbose_eval=25, feval=gini_lgb, early_stopping_rounds=70)    
        #_____________________________
        # Sum predictions on test data
        temp = mdl.predict(X_test, num_iteration = mdl.best_iteration)
        predictions = temp if j==0 else temp+predictions
        #________________________________
        # Predict on our validation data
        predict_valid = mdl.predict(x_valid, num_iteration = mdl.best_iteration )
        #_______________________
        # save these predictions
        prediction_valid_by_fold = pd.DataFrame()
        list_valid_index = list(map(int, id_train[valid_index]))        
        prediction_valid_by_fold['id']     = list(map(int, id_train[valid_index]))
        prediction_valid_by_fold.set_index(valid_index, inplace = True)
        prediction_valid_by_fold.loc[valid_index, 'target'] = predict_valid 
        fichier = path_model[num_mod]+'_pred_valid_fold_{}-{}.csv'.format(j+1, kfold)
        prediction_valid_by_fold.to_csv(fichier, index=False)    
    #___________________________________________________________________________
    prediction_test = pd.DataFrame()
    prediction_test['id'] = list(map(int, id_test))
    prediction_test['target'] = predictions / kfold
    fichier = path_model[num_mod]+'_pred_test.csv'
    prediction_test.to_csv(fichier, index=False)

print("--- elapsed time {} seconds ---".format(time.time() - start_time))
#=========================================================
# IV. Load LGBM models results
#=========================================================
def read_folds(nfolds, fichier):
    df_model = pd.DataFrame()
    for j in range(nfolds):    
        fichier_fold = fichier + '_pred_valid_fold_{}-{}.csv'.format(j+1, nfolds)
        df = pd.read_csv(fichier_fold)
        df_model = pd.concat([df, df_model])
    df_model.sort_values('id', inplace = True)
    df_model = df_model.reset_index(drop = True)
    return df_model
#__________________________________________
df_model = []
for model_directory in path_model:    
    df_model.append(read_folds(kfold, model_directory))  
#__________________________________________
print("Individual gini's scores:")
for i, model in enumerate(path_model):
    print('{:<20} -> {:8.5f}'.format(model, gini_normalized(y_train,
           list(df_model[i]['target']) )))
#=========================================================
# V. Determine optimal weights
#=========================================================
coefficients = np.zeros((kfold, NB_models))
valid_sum = 0
#_______________________________________________________________________
sss = StratifiedKFold(n_splits = kfold, shuffle = True, random_state=33)
for j, (trn_idx, vld_idx) in enumerate(sss.split(y_train, y_train)): 
    print(50*"-"+'\n[Fold {}/{}]'.format(j + 1, kfold))
    target_train = y_train[trn_idx]
    target_valid = y_train[vld_idx]
    #____________________
    train, valid = [], []
    for k, model in enumerate(path_model):
        train.append(df_model[k].loc[trn_idx])
        valid.append(df_model[k].loc[vld_idx])
    #____________________
    def function_gini(x):
        x = x / np.linalg.norm(x)
        x = [max(val, 0) for val in x]
        y_test = np.zeros(len(target_train))
        for k in range(NB_models):
            y_test += x[k]*np.array(train[k]['target'])
            
        fct = gini_normalized(target_train, y_test)
        return -fct
    #_________________________________
    x0 = [1 for _ in range(NB_models)]
    
    res = minimize(function_gini, x0, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
    coeffs = [max(val, 0) for val in res.x]
    coefficients[j,:] = coeffs / np.linalg.norm(coeffs)
    print('\nfinal vector: {}'.format(coefficients[j,:]))
        
    y_test = np.zeros(len(target_valid))
    for k in range(NB_models):
        y_test += coefficients[j, k]*np.array(valid[k]['target'])
                
    valid_sum += gini_normalized(target_valid, y_test)
    print("valid's gini: {:<7.4f} \n".format(gini_normalized(target_valid, y_test)))
#____________________________________________________________________
print(50*'='+"\nOverall's gini: {:<8.5f}".format(valid_sum / kfold))
#=========================================================
# VI. Create submission file
#=========================================================
print(50*'-' + '\n Initial set of weigths: \n', coefficients)
t = [coefficients[:,j].mean() for j in range(NB_models)]
print(50*'-' + "\n Weights averaged over folds: \n {}".format(t) )
#_____________________________
y_test = np.zeros(len(y_train))
for k in range(NB_models):
    y_test += t[k]*np.array(df_model[k]['target'])
#_____________________________________________________________________________________________________
print("\n valid's gini with mean weights =====> {:<8.6f} \n".format(gini_normalized(y_train, y_test)))
#___________________________________
sub = [_ for _ in range(NB_models)]
for j in range(NB_models):
    fichier = path_model[j]+'_pred_test.csv'.format(j+1, kfold)
    sub[j] = pd.read_csv(fichier)
#__________________________
id_test = sub[0]['id']
avg = np.zeros(len(id_test))
for i in range(NB_models):
    avg += t[i] * sub[i]['target'] 
#______________________    
submit = pd.DataFrame()
submit['id'] = id_test
submit['target'] = avg 
submit.to_csv('submission.csv', index=False)


    