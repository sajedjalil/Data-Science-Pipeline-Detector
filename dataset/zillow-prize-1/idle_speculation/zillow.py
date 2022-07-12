import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from keras.models import Model
from keras.layers import Input, Dense

num_fit=10
k=[ 0.00315726,  0.82076004,  0.12516465]

train_2016 = pd.read_csv("../input/train_2016_v2.csv")
train_2017 = pd.read_csv("../input/train_2017.csv")
properties_2017 = pd.read_csv("../input/properties_2017.csv", low_memory=False)

train_all=pd.concat([train_2016,train_2017])
train_all['transactionmonth']=train_all['transactiondate'].apply(lambda x: int(x[5:7]))

for c in properties_2017.columns[1:]:
    z=properties_2017[c].copy()
    if z.dtype=='object':
        z=pd.factorize(z)[0]
    z[np.isnan(z)]=-999999
    z=rankdata(z,method='dense')
    z-=z.min()
    z=z*1./z.max()
    properties_2017[c]=z.astype(np.float32)

train_properties=properties_2017.merge(train_all,on='parcelid')
y=train_properties['logerror']
train_properties.drop(['parcelid','transactiondate','logerror'],axis=1,inplace=True)
train_properties['transactionmonth']=train_properties['transactionmonth']*1./12
Z=train_properties.values.astype(np.float32)

parcel_index=properties_2017['parcelid']
W=properties_2017
W.drop('parcelid',axis=1,inplace=True)
W['transactionmonth']=0.
W=W.values.astype(np.float32)

dlrn = lgb.Dataset(Z, label=y)
params = {}
params['max_bin'] = 63
params['learning_rate'] = 0.01
params['objective'] = 'regression_l1'
params['metric'] = 'l1'
params['feature_fraction'] = 0.90
params['bagging_fraction'] = 0.85
params['bagging_freq'] = 1
params['min_data'] = 500
params['verbose'] = 0
params['feature_fraction_seed'] = 0
params['data_random_seed']=0
params['num_threads']=4


lgb_test_preds=[]

for j in range(3):
    lgb_test_preds.append(np.zeros(W.shape[0]))

for i in range(num_fit):    
    print("lgb: "+str(i))
    tstart=time.time()
    np.random.seed(i)
    params['bagging_seed'] = i
    evals_dict=dict()
    m = lgb.train(
                    params
                    ,dlrn
                    ,num_boost_round=3200
                    ,verbose_eval=False
                 )
    for j in range(3):
        W[:,-1]=(j+10)*1./12
        lgb_test_preds[j]+=m.predict(W)*1./num_fit
    tend=time.time()
    print(tend-tstart)

def get_net():
    a=Input(shape=(Z.shape[1],))
    b=Dense(256,activation='relu')(a)
    b=Dense(32,activation='relu')(b)
    b=Dense(1)(b)
    nn=Model(inputs=a,outputs=b)
    return(nn)

nn_test_preds=[]

for j in range(3):
    nn_test_preds.append(np.zeros(W.shape[0]))

for i in range(num_fit):
    print('nn: '+str(i))
    tstart=time.time()
    np.random.seed(i)
    nn=get_net()
    nn.compile(optimizer='adagrad',loss='mean_absolute_error')
    nn.fit(Z, y, epochs=15, batch_size=32, verbose=0)
    for j in range(3):
        W[:,-1]=(j+10)*1./12
        nn_test_preds[j]+=nn.predict(W, batch_size=4096).reshape((W.shape[0]))*1./num_fit
    tend=time.time()
    print(tstart-tend)

fnl_pred=np.zeros((W.shape[0],3))

for j in range(3):
    fnl_pred[:,j]=k[0]+k[1]*lgb_test_preds[j]+k[2]*nn_test_preds[j]

sub=pd.DataFrame(np.hstack([fnl_pred,fnl_pred]))
sub=pd.concat([parcel_index,sub],axis=1)
sub.columns=['parcelid','201610','201611','201612','201710','201711','201712']

sub.to_csv('sub_01.csv',index=False)
