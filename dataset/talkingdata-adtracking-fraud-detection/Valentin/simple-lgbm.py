import pandas as pd
import numpy as np
import gc
import time
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
        
### Function
def conversion (df, var):
    if df[var].dtype != object:
        if abs(sum(df[var] - df[var].astype(np.uint64))) < 0.0001:
            maxi = df[var].max()
            if maxi < 255:
                df[var] = df[var].astype(np.uint8)
                print(var,"converted to uint8")
            elif maxi < 65535:
                df[var] = df[var].astype(np.uint16)
                print(var,"converted to uint16")
            elif maxi < 4294967295:
                df[var] = df[var].astype(np.uint32)
                print(var,"converted to uint32")
            else:
                df[var] = df[var].astype(np.uint64)
                print(var,"converted to uint64")
        else :
                df[var] = df[var].astype(np.float32)
                print(var, "converted to float32")

### Parameters setting  ###
nrows = 50 * 10 **6
nround = 800
params = {
    'boosting_type': 'gbdt',
    'objective':'binary',
    'num_leaves': 7,
    'max_bin' : 100,
    'subsample_for_bin': 200000,
    'max_depth' : 3,
    'objective': 'binary',
    'min_data_in_leaf': 100,
    'learning_rate': 0.1,
    'subsample' : 0.7,
    'subsample_freq': 1,
    'metric': 'auc',
    'num_threads': 4,
    'scale_pos_weight':200,
    'save_binary':True,
    'min_split_gain': 0, 
    'reg_alpha': 0,  
    'reg_lambda': 0, 
}                
Apply_to_test = True

### Train importation ###
start_time = time.time()
print("Importation is running")

p1 = pd.read_csv('../input/feature-engineering-apply-to-train/p1.csv', nrows = nrows, 
                    usecols = ['channel', 'freq_ip', 'is_attributed', 'hour', 'app', 'os'], compression = "gzip")
[conversion (p1, v) for v in p1]
p2 = pd.read_csv('../input/feature-engineering-apply-to-train/p2.csv', nrows = nrows, 
                    usecols = ['clicks_by_device', 'last_click', 'channel_by_ip', 'app_by_use', 'nextClick'], compression = "gzip")
[conversion (p2, v) for v in p2]
p3 = pd.read_csv('../input/feature-engineering-apply-to-train/p3.csv', nrows = nrows, 
                    usecols = ['dl_rate_app', 'dl_rate_os', 'app_by_ip', 'ip_app_os_count', 'ip_app_count'], compression = "gzip")
[conversion (p3, v) for v in p3]
#p4 = pd.read_csv('../input/feature-engineering-apply-to-train-p2/p4.csv', nrows = nrows, usecols = ['nip_hh_dev', 'nip_hh_app', 'nip_hh_os', 'nip_test_hh'], compression = "gzip")
#[conversion (p4, v) for v in p4]

train = pd.concat([p1, p2, p3], axis = 1)
train.sort_index(axis = 1, inplace = True)
del p1, p2, p3
gc.collect()
    
print("Importation completed in ", round((time.time()-start_time)/60,0),"minutes")

### Data preparation ###
start_time = time.time()
print("Data preparation before modelisation is running")

y = train['is_attributed']
train.drop('is_attributed', axis = 1, inplace = True)

list_vars = list(train.columns)
list_vars.append("click_id")

x1, x2, y1, y2 = train_test_split(train, y, test_size = 0.1, random_state = 1)
del y, train
gc.collect()

dtrain = lgb.Dataset(x1, label=y1, categorical_feature = ['channel', 'hour', 'app', 'os'])
dval = lgb.Dataset(x2, label=y2, reference=dtrain, categorical_feature = ['channel', 'hour', 'app', 'os'])
del x1, x2, y1, y2
gc.collect()

print("Data preparation completed :", round(time.time()-start_time,0),"seconds")

### LGBoost training ###
start_time = time.time()
print(" LGBM is training")
print(" We have ", len(list_vars) -1,"features")

model = lgb.train(params, dtrain, num_boost_round=nround, valid_sets=[dtrain, dval], early_stopping_rounds=30, verbose_eval=10)

del dtrain, dval
gc.collect()          
print("LGBM train completed :", round(time.time()-start_time,0)/60,"minutes")

### Model evaluation ###
features_importance = pd.DataFrame(list(zip(model.feature_name(), model.feature_importance())))
features_importance.columns = ['name', 'importance']
features_importance.sort_values('importance', inplace = True, ascending = False)
features_importance.set_index('name', inplace = True)
print("Feature's contributions :\n", features_importance)

#ax = lgb.plot_importance(model, max_num_features=100)
#plt.show()
#fig.savefig('feature_importance_v11.png')

### Model application to test ### 
if Apply_to_test :
    start_time = time.time()
    print(" The model is applying to the test file")
    
    test = pd.read_csv('../input/feature-engineering-apply-to-test/test_up.csv', \
        usecols = list_vars, \
        compression = 'gzip')
    [conversion(test, v) for v in test]
    test.sort_values('click_id', inplace = True)
    
    sub=pd.DataFrame()
    sub['click_id'] =  test['click_id']
    test.drop('click_id', axis = 1, inplace = True)
    test.sort_index(axis = 1, inplace = True)
    
    sub['is_attributed'] = model.predict(test, num_iteration=model.best_iteration)
    
    order = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv', usecols = [0])
    order['order'] = range(1, len(order) + 1)
    sub = pd.merge(sub, order , on = 'click_id', how = 'right')
    sub = sub.sort_values('order').drop('order', axis = 1)
    
    print(" Model application is completed :", round(time.time()-start_time,0)/60,"minutes")
    
    ### Exportation ###
    sub.to_csv('my_lgbm.csv', index = False, float_format='%.9f')