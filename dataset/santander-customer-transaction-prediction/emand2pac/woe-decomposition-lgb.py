# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import warnings
import lightgbm as lgb
from sklearn import preprocessing
from sklearn import decomposition

warnings.filterwarnings('ignore')

PATH = '../input/'
train_raw=pd.read_csv(PATH+'train.csv',low_memory=False)

print(train_raw.shape)

target=train_raw['target']
#train.describe().to_excel('val_describe.xlsx')

pos_neg_rate=train_raw['target'].value_counts()
#woe_cols=[81,139,53,26,12,26,99,2,44,109]
woe_cols=range(2,202)

def _woe(data):
    #woe_df_bins=pd.DataFrame()
    woe_df_score=[]
    for i in woe_cols:
        #print(i)
        data['woe'+str(i)]=pd.qcut(data.iloc[:,i],10,labels=range(0,10))
        woe_df=pd.crosstab(data['woe'+str(i)],data['target'])
        #woe_df_bins['bins'+str(i)]=woe_df.index
        woe_df['woe_score'+str(i)]=np.log((woe_df.iloc[:,1]/woe_df.iloc[:,0])/(pos_neg_rate[1]/pos_neg_rate[0])) 
        #woe_df_score['score'+str(i)]=woe_df['woe_score'+str(i)].values
        '''
        plt.figure()
        plt.bar(x=woe_df.index,height=woe_df['woe_score'+str(i)])
        plt.title('woe_score'+str(i))
        plt.show()
        '''
        woe_df_score.append(woe_df['woe_score'+str(i)])
        data=pd.merge(data,pd.DataFrame(woe_df['woe_score'+str(i)]),how='left',left_on='woe'+str(i),right_index=True)
        data.drop('woe'+str(i),axis=1,inplace=True)
    return data,woe_df_score
train_raw,woe_df_score=_woe(train_raw)
train=train_raw.drop(['ID_code','target'],axis=1)  

scaler=preprocessing.StandardScaler()

def _extract_features(data):
    data['std']=data.iloc[:,0:200].std(axis=1)
    data['mean']=data.iloc[:,0:200].mean(axis=1)
    data['skew']=data.iloc[:,0:200].skew(axis=1)
    data['sum']=data.iloc[:,0:200].sum(axis=1)
    data['max']=data.iloc[:,0:200].max(axis=1)
    data['kurt']=data.iloc[:,0:200].kurtosis(axis=1)
    data['median']=data.iloc[:,0:200].median(axis=1)
    for col in ['std','mean','skew','sum','max','kurt','median']:
        data[col]=scaler.fit_transform(data[col].values.reshape(-1,1))
    return data
train=_extract_features(train)


n_com=1

def _feature_engineering(data,n_com):
    
    #feature_names=data.columns.values
    print('-----scaler-------')
    
    data_scaler=scaler.fit_transform(data.iloc[:,0:200])
    #data=pd.DataFrame(data,columns=feature_names)
    data.iloc[:,0:200]=data_scaler
    
    print('-----decomposition-------')
    tsvd=decomposition.TruncatedSVD(n_components=n_com,random_state=15)
    data_tsvd=tsvd.fit_transform(data_scaler)
    #data_tsvd=scaler.fit_transform(data_tsvd)
    
    pca=decomposition.PCA(n_components=n_com,random_state=15)
    data_pca=pca.fit_transform(data_scaler)
    #data_pca=scaler.fit_transform(data_pca)
    
    ica=decomposition.FastICA(n_components=n_com,random_state=15)
    data_ica=ica.fit_transform(data_scaler)
    #data_ica=scaler.fit_transform(data_ica)
    
    fa=decomposition.FactorAnalysis(n_components=n_com,random_state=15)
    data_fa=fa.fit_transform(data_scaler)
    #data_fa=scaler.fit_transform(data_fa)
    
    for i in range(0,n_com):
        data['tsvd_'+str(i)]=data_tsvd[:,i]
        data['pca_'+str(i)]=data_pca[:,i]
        data['ica_'+str(i)]=data_ica[:,i]
        data['fa_'+str(i)]=data_fa[:,i]   
    return data

data=_feature_engineering(train,n_com)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=1)
lda.fit(data,target)
data['lda']=lda.transform(data)

test_raw=pd.read_csv(PATH+'test.csv',low_memory=False)

test_id=pd.DataFrame(test_raw['ID_code'])
test_raw.drop('ID_code',axis=1,inplace=True)
woe_cols=range(0,200)
for col,df in zip(woe_cols,woe_df_score):
    test_raw['woe'+str(col)]=pd.qcut(test_raw['var_'+str(col)],10,labels=range(0,10))
    test_raw=pd.merge(test_raw,pd.DataFrame(df),how='left',left_on='woe'+str(col),right_index=True)
    test_raw.drop('woe'+str(col),axis=1,inplace=True)  
    
test_raw=_extract_features(test_raw)
test=_feature_engineering(test_raw,n_com)
test['lda']=lda.transform(test)


param = {
    'bagging_freq': 5,          'bagging_fraction': 0.33,   'boost_from_average':'false',   'boost': 'gbdt',
    'feature_fraction': 0.045,   'learning_rate': 0.008,     'max_depth': -1,                'metric':'auc',
    'min_data_in_leaf': 80,     'min_sum_hessian_in_leaf': 10.0,'num_leaves': 13,           'num_threads': 8,
    'tree_learner': 'serial',   'objective': 'binary',      'verbosity': 1
    }

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
num_round = 20000
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=12345)
oof = np.zeros(len(data))
predictions = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(data.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(data.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(data.iloc[val_idx], label=target.iloc[val_idx])
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 4000)
    oof[val_idx] = clf.predict(data.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


test_id["target"] = predictions
from datetime import datetime
now=datetime.now()
test_id.to_csv('submission'+str(now.month)+'month'+str(now.day)+'day'+str(now.hour)+'hour'+str(now.minute)+'minute'+str(round(roc_auc_score(target, oof),4))+'.csv',index=False)
