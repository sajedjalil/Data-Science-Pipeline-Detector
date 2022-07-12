# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# -----------------------------------------------------------------------------
# LightGBM regression example
# __author__ = "DDgg"
# https://www.kaggle.com/c/mercedes-benz-greener-manufacturing
# -----------------------------------------------------------------------------

import numpy as np
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# measure time
import time
start = time.time()

# PCA -------------------------------------------------------------------------
# add to see skewness
# -----------------------------------------------------------------------------

# data imnport 
# fork of forks from https://www.kaggle.com/jaybob20/starter-xgboost
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#pca_3D_plot(test)

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] =  lbl.transform(list(test[c].values))
        
#------------------------------------
# pca and Ica transform from https://www.kaggle.com/uluumy/mercedez-baseline-2
from sklearn.decomposition import PCA, FastICA

# define number of principal components for PCA and ICA
n_comp = 15

# PCA  // Principal component analysis
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA // Independent component analysis
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]


y_train = train["y"]
y_mean = np.mean(y_train)
train.drop('y', axis=1, inplace=True)

#------------------------------------

# best by RMSE coming from MB-loop-320k PCA/ICA
# best_rnd = [297945,274899,310464,289355,101978,16350]

# define best by R2 MB-loop-320k PCA/ICA
# best_rnd = [297945,274899,310464,289355,101978,16350]

# -----------------------------------
# baseline 6 PCA and ICA
# Average R2: 0.687282898109 average RMSE: 6.40002419534
# Average R2: 0.691423775211 average RMSE: 6.35759672637 //  'num_leaves': 5,
# Average R2: 0.691697545438 average RMSE: 6.35506564301 //  'min_data_in_leaf':10,
# Average R2: 0.692050622747 average RMSE: 6.35137639156 // 'min_data_in_leaf':8,
# Average R2: 0.692708293942 average RMSE: 6.34441921967 // 'feature_fraction': 0.4,
# Average R2: 0.693566750002 average RMSE: 6.33541869378 // 'feature_fraction': 0.4, n_comp = 20
# Average R2: 0.693664255971 average RMSE: 6.33443572002 // 'feature_fraction': 0.4, n_comp = 15

# diff merger 33 best
best_rnd =[280676,302520,63466,264360,56994,25207,297945,127781,307846,157015,75580,247034,115661,145050,298917,248847,310464,74984,155191,77004,315578,263139,181108,138577,14316,289355,27064,201580,55409,274899,93907,263812]

max_loop = len(best_rnd)

# create empty dataframe
df1 = pd.DataFrame(index=np.arange(0, max_loop),columns=['ID', 'biter','r2','rmse'])
df2 = pd.DataFrame(index=np.arange(0, 4209))
# loop through the best RMSE and R2
for i, value in enumerate(best_rnd):
    # print (i, value)

    X_train_new, X_valid_new, y_train_new, y_valid_new = \
    train_test_split(train, y_train, test_size=0.2, random_state=value)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train_new, y_train_new)
    lgb_valid = lgb.Dataset(X_valid_new, y_valid_new, reference=lgb_train)

    # to record eval results for plotting
    evals_result = {} 

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'max_bin' : 50,
        'num_leaves': 5,            # STD  10
        'learning_rate': 0.06,      # STD  0.06
        'feature_fraction': 0.4,    # STD  0.9
        'bagging_fraction': 0.85,   # STD  0.85
        'bagging_freq': 4,          # STD  4 
        'min_data_in_leaf':8,       # STD  4
        'min_sum_hessian_in_leaf': 10, # STD 0.8
        'verbose':0,
        'num_threads':8 # 2:(23s) 4:(18s) 8:(15s) 16:(17s), 32:(34s)
    }

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,
                    valid_sets=[lgb_train, lgb_valid],
                    evals_result=evals_result,
                    verbose_eval=0,
                    early_stopping_rounds=40)

    # predict
    y_pred = gbm.predict(X_valid_new, num_iteration=gbm.best_iteration)
    
    
    # eval r2-score 
    from sklearn.metrics import r2_score
    r2 = r2_score(y_valid_new, y_pred)

    # eval rmse loop number, stopping rounds, rmse
    print(i,"  ",gbm.best_iteration,"  ",r2,"  ", mean_squared_error(y_valid_new, y_pred) ** 0.5)
    df1.set_value(i,'ID',i) # index
    df1.set_value(i,'biter',gbm.best_iteration) #GBM iteration
    df1.set_value(i,'r2',r2) # r2
    df1.set_value(i,'rmse',mean_squared_error(y_valid_new, y_pred) ** 0.5) # RMSE
 
    # Predicting test set...
    y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)
    df2[value] = y_pred
   
    # ------- end loop
end = time.time()
print ("Time: ",round(end - start,2), "sec")

# print average R2 and RMSE
print ("Average R2:",  df1.mean(axis = 0)['r2'],"average RMSE:",  df1.mean(axis = 0)['rmse'])


# create average of all
y_average = df2.mean(axis = 1 )

# write test set with average of all
output = pd.DataFrame({'id': test['ID'], 'y': y_average})
output.to_csv('submit-lightgbm-5-average.csv', index=False)

# write loop test RMSE and R2
# df1.to_csv('best-10-rmse.csv')      
print("finished.")
