import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
##################################################################################
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


print(os.listdir("../input"))
os.chdir("../input")
DATA_DIR = "../input"




#print(os.cwd())
# Data drectory
#DATA_DIR = "/kaggle/input/bosch-production-line-performance/"

train_date = os.path.join(DATA_DIR,'train_date.csv')
test_num= os.path.join(DATA_DIR,'test_numeric.csv')
train_num= os.path.join(DATA_DIR,'train_numeric.csv')
test_date = os.path.join(DATA_DIR,'test_date.csv')
print(train_date)

ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'


CHUNKSIZE = 50000
NROWS = 250000

# Feauture from faron kernel
feat =['L1_S24_F1723',
 'L3_S30_F3809',
 'L3_S32_F3850',
 'L3_S30_F3519',
 'L3_S29_F3407',
 'L3_S33_F3855',
 'L3_S33_F3859',
 'L3_S38_F3952',
 'L3_S33_F3865',
 'L1_S24_F1846',
 'L1_S24_F1632',
 'L1_S24_F1695',
 'L3_S29_F3336',
 'L1_S24_F1604',
 'L3_S29_F3324',
 'L3_S29_F3351',
 'L3_S34_F3876',
 'L3_S29_F3461',
 'L3_S29_F3373',
 'L2_S26_F3121',
 'L3_S29_F3357',
 'L3_S30_F3569',
 'L1_S24_F1838',
 'L3_S29_F3342',
 'L0_S0_F20',
 'L1_S24_F1565',
 'L3_S29_F3330',
 'L3_S30_F3544',
 'L3_S30_F3804',
 'L0_S0_F18']

print('Magic feauture')
train = pd.read_csv(train_num,usecols=['Id'])
test = pd.read_csv(test_num,usecols=['Id'])



nrows = 0
for tr, te in zip(pd.read_csv(train_date, chunksize=CHUNKSIZE), pd.read_csv(test_date, chunksize=CHUNKSIZE)):
    feats = np.setdiff1d(tr.columns, ['Id'])

    stime_tr = tr[feats].min(axis=1).values
    stime_te = te[feats].min(axis=1).values

    train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
    test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te
    

# try faron magic feuture
n= train.shape[0]
x= pd.concat([train,test]).reset_index()
x['id1']= x['Id'].diff()
x['id2']=x['Id'][::-1].diff()

x = x.sort_values(by=['StartTime','Id'],ascending=True)

x['id3']=x['Id'].diff()
x['id4']=x['Id'][::-1].diff()

train =x.sort_index().iloc[:n,:]
test = x.sort_index().iloc[n:,:]

# Read numeric data with importance features
print('reading data')
train_data = pd.DataFrame()
test_data = pd.DataFrame()
for x in pd.read_csv(train_num, chunksize=CHUNKSIZE,usecols=feat+['Response','Id']):
    train_data=pd.concat([train_data,x])
for y in pd.read_csv(test_num,chunksize=CHUNKSIZE,usecols=feat+['Id']):
    test_data=pd.concat([test_data,y])
    
    
    

train_data[['startime','id1','id2','id3','id4']]= train[['StartTime','id1','id2','id3','id4']].fillna(9999999)
test_data[['startime','id1','id2','id3','id4']]= test[['StartTime','id1','id2','id3','id4']].reset_index(drop=True).fillna(9999999)

train_x=train_data.drop(['Response','startime'],axis=1).values
train_y= train_data['Response'].values
test_x = test_data.drop(['startime'],axis=1).values

#Model evaluate

print('model validating')
def Mcc(y_true,y_pred):
    tn= confusion_matrix(y_true, y_pred)[0, 0]
    fp= confusion_matrix(y_true, y_pred)[0, 1]
    fn= confusion_matrix(y_true, y_pred)[1, 0]
    tp= confusion_matrix(y_true, y_pred)[1, 1]
    if ((tn+fn)*(tn+fp)*(tp+fn)*(tp+fp))>0:
        return (tn*tp-fn*fp)/((tn+fn)*(tn+fp)*(tp+fn)*(tp+fp))**0.5 
    else: return 0
    
    
def train(train_x,train_y):
    dtrain = xgb.DMatrix(train_x,label=train_y)

    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'max_depth': 4,
        'num_parallel_tree': 1,
        'min_child_weight': 2,
        'eval_metric': 'auc',
        'base_score': 0.00581120}#parameter from XGB baseline model

    num_round=10
    bst = xgb.train(xgb_params, dtrain, )
    return bst


print('training')
xtrain,xtest,ytrain,ytest= train_test_split(train_x,train_y,test_size=0.3)
bst= train(xtrain,ytrain)
ypred=bst.predict(xgb.DMatrix(xtest))
threshold=np.linspace(0.01,0.99,50)
mcc=np.array([Mcc(ytest,np.where(ypred>thres,1,0)) for thres in threshold])
plt.plot(threshold,mcc)
    

# best threshold
best_thres=threshold[mcc.argmax()]


# fit data to model
bst2 = train(train_x,train_y)
ypred2 = bst2.predict(xgb.DMatrix(test_x))

my_submission = test_data[['Id']].merge(pd.DataFrame(np.where(ypred2>best_thres,1,0)),left_index=True,right_index=True)
my_submission.columns = ['Id', 'Response']
my_submission.to_csv(os.path.join('/kaggle/working','submission.csv'),index=False)