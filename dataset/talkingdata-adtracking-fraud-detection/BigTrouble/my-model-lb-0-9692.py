import pandas as pd
import numpy as np
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

start = datetime.now()
VALIDATE = False
RANDOM_STATE = 50
VALID_SIZE = 0.90
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 650
skiprows = range(1,179903891)
nrows = 5000000
output_filename = 'firstresult.csv'

path = '../input/'

dtypes = {'ip':'uint32',
          'app':'uint16',
          'device':'uint16',
          'os':'uint16',
          'channel':'uint16',
          'is_attributed':'uint8',
          'click_':'uint32'}
id
gc.collect()
train_cols = ['ip','app','device','os','channel','click_time','is_attributed']
print('reading data')
train_df = pd.read_csv(path+'train.csv',skiprows = skiprows,nrows = nrows,dtype = dtypes,usecols = train_cols)
print('complete read data')
gc.collect()
len_train = len(train_df)
gc.collect()
most_freq_hours_in_test_data = [4,5,9,10,13,14]
least_freq_hours_in_test_data = [6,11,15]

def prep_data( df ):
    
    print("1")
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    print("complete 1")
    print("2")
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    print("complete 2")
    print("3")
    df.drop(['click_time'], axis=1, inplace=True)
    print("complete 3")
    gc.collect()
    print("4")
    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    print("complete 4")
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day', 'in_test_hh'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_test_hh'})
    print("5")
    df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
    print("complete 5")
    print("6")
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
    print("complete 6")
    del gp
    gc.collect()
    print("7")
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_hh'})
    print("complete 7")
    print("8")
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
    print("complete 8")
    del gp
    gc.collect()
    print("9")
    gp = df[['ip', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_os'})
    df = df.merge(gp, on=['ip','os','hour'], how='left')
    df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
    print("complete 9")
    del gp
    gc.collect()
    print("10")
    gp = df[['ip', 'app', 'hour', 'channel']].groupby(by=['ip', 'app',  'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_app'})
    df = df.merge(gp, on=['ip','app','hour'], how='left')
    df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
    print("complete 10")
    del gp
    gc.collect()
    print("11")
    gp = df[['ip', 'device', 'hour', 'channel']].groupby(by=['ip', 'device', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_dev'})
    df = df.merge(gp, on=['ip','device','hour'], how='left')
    df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
    print("complete 11")
    del gp
    gc.collect()

    df.drop( ['ip','day'], axis=1, inplace=True )
    gc.collect()
    return df

train_df = prep_data(train_df)
gc.collect()


# In[23]:


test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=test_cols)
test_df = prep_data(test_df)
clickid = test_df['click_id']
gc.collect()


# In[26]:


ntrain =train_df.shape[0]
ntest = test_df.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(ntrain,n_folds=NFOLDS,random_state=SEED)
class SklearnHelper(object):
    def __init__(self,clf,seed=0,params=None):
        params['random_state'] = seed
        self.clf=clf(**params)
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
    def predict(self,x):
        return self.clf.predict(x)
    def fit(self,x,y):
        return self.clf.fit(x,y)
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)


# In[27]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[28]:


#Random Forest parameter
rf_params = {
    'n_jobs':-1,
    'n_estimators':500,
    'warm_start':True,
    'max_depth':6,
    'min_samples_leaf':2,
    'max_features':'sqrt',
    'verbose':0
}
#Extra Trees Parameters
et_params = {
    'n_jobs':-1,
    'n_estimators':500,
    'max_depth':8,
    'min_samples_leaf':2,
    'verbose':0
}
#Adaboost parameters
ada_params = {
    'n_estimators':500,
    'learning_rate':0.75
}
#Gradient Boosting parameters
gb_params = {
    'n_estimators':500,
    'max_depth':5,
    'min_samples_leaf':2,
    'verbose':0
}
#SVM parametors
svc_params = {
    'kernel':'rbf',
    'C':0.025
}


# In[29]:


rf = SklearnHelper(clf=RandomForestClassifier,seed=SEED,params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier,seed=SEED,params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier,seed=SEED,params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier,seed=SEED,params=gb_params)
svc = SklearnHelper(clf=SVC,seed=SEED,params=svc_params)


# In[30]:


y_train_df = train_df['is_attributed']
x_train_df = train_df.drop(['is_attributed'],axis=1)
x_train_df = train_df.values
x_test_df = test_df.values


# In[34]:


print("et begin")
et_oof_train,et_oof_test = get_oof(et,x_train_df,y_train_df,x_test_df)
print("rf begin")
rf_oof_train,rf_oof_test = get_oof(rf,x_train_df,y_train_df,x_test_df)
print("ada begin")
ada_oof_train,ada_oof_test = get_oof(ada,x_train_df,y_train_df,x_test_df)
print("gb begin")
gb_oof_train,gb_oof_test = get_oof(gb,x_train_df,y_train_df,x_test_df)
print("svm begin")
svc_oof_train,svc_oof_test = get_oof(svc,x_train_df,y_train_df,x_test_df)
print("Training is complete!")

x_train_df = np.concatenate((et_oof_train,rf_oof_train,ada_oof_train,gb_oof_train,svc_oof_train),axis=1)
x_test_df = np.concatenate((et_oof_test,rf_oof_test,ada_oof_test,gb_oof_test,svc_oof_test),axis=1)
x_train_df.to_csv("firsttrain.csv")
x_test_df.to_csv("firsttest.csv")
gc.collect()
dtrain = lgb.Dataset(x_train_df.values, label=y_train_df.values)
del x_train_df
gc.collect()

evals_results = {}
print("begin training")
model = lgb.train(params, 
                  dtrain, 
                  valid_sets=[dtrain], 
                  valid_names=['train'], 
                  evals_result=evals_results, 
                  num_boost_round=OPT_ROUNDS,
                  verbose_eval=50,
                  feval=None)
print("complete training")
del dtrain
gc.collect()
sub = pd.DataFrame()
sub['click_id'] = clickid
sub['is_attributed'] = model.predict(x_test_df)
sub.to_csv(output_filename, index=False, float_format='%.9f')
print(datetime.now(), '\n')
print('{:^17} : {:}'.format('train time', datetime.now()-start))
