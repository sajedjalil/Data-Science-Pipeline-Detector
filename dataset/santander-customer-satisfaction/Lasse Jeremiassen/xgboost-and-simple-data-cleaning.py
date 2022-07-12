from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
# from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import xgboost as xgb
import csv
import sys

df_train=pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")
test_id = df_test['ID']

df_train.head()
df_train.columns

del df_train['ID']
del df_test['ID']
yy=df_train.TARGET.copy(deep=True)
del df_train['TARGET']
df_train['n0']=df_train.sum(axis=1)
df_train.loc[df_train['n0']==0,'n0']=True
df_train.loc[df_train['n0']!=0,'n0']=False
df_test['n0']=df_test.sum(axis=1)
df_test.loc[df_test['n0']==0,'n0']=True
df_test.loc[df_test['n0']!=0,'n0']=False


for clm_nm in df_train.columns:
    if(len(df_train.loc[:,clm_nm].unique())<2):
        if clm_nm in df_train.columns:
            del df_train[clm_nm]
        if clm_nm in df_test.columns:
            del df_test[clm_nm]
remove=[]
for i in range(len(df_train.columns)):
    for j in range(i+1,len(df_train.columns)):
        clm = df_train.columns[i]
        clm1 = df_train.columns[j]
        if df_train[clm].equals(df_train[clm1]) and not clm in remove:
            remove += [clm]
for nm in remove:
    if nm in df_train.columns:
        del df_train[nm]
    if nm in df_test.columns:
        del df_test[nm]
cat_able=[]
k=0
for nm in df_train.columns:
    if not nm in df_test.columns:
        continue
    max_v = np.max(df_train.loc[:,nm])
    min_v = np.min(df_train.loc[:,nm])
    k+=1
    if max_v - min_v <=500 and max_v>0 and min_v>0:
        cat_able+=[k]
    df_test.loc[df_test[nm]>max_v,nm]=max_v
    df_test.loc[df_test[nm]<min_v,nm]=min_v

#dft_con = df_train.copy(deep=True)
#dft_cat = df_train[cat_able].copy(deep=True)
#for nm in cat_able:
#    del dft_con[nm]

#
enc = OneHotEncoder(categorical_features = cat_able)
train = enc.fit_transform(df_train)
test = enc.transform(df_test)
xgb = XGBClassifier(max_depth=5, learning_rate=0.0202048, n_estimators=560, objective='binary:logistic', subsample=0.6815, colsample_bytree=0.701, seed=1234)
xgb.fit(train, yy)
y_pred_xgb = xgb.predict_proba(test)
result = y_pred_xgb[:,1]
result=zip(test_id.values, result)
predictions_file = open("submission.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['ID','TARGET'])
open_file_object.writerows(result)
predictions_file.close()
#dtrain = xgb.DMatrix(train)
#evallist  = [(dtrain,'train')]
#param = {'bst:max_depth':5, 'bst:eta':0.0202048, 'silent':1, 'objective':'binary:logistic',
#         'booster': "gbtree",'subsample': 0.6815,'colsample_bytree': 0.701, }
#param['eval_metric'] = 'auc'
#num_round = 560         
#bst = xgb.train( param, dtrain, num_round, evallist,verbose_eval=True )
#test = xgb.DMatrix(enc.transform(df_test))
#y_tesst = bst.predict(test)
