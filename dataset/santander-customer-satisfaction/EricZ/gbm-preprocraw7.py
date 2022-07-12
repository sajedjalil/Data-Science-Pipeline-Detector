import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import patsy
import xgboost as xgb
import sklearn as sk
import pickle
import datetime 

## read csv
trainX= pd.read_csv('../input/train.csv')
trainY= trainX['TARGET']
del trainX['TARGET']
testX= pd.read_csv('../input/test.csv')
combX= pd.concat((trainX, testX), axis=0, ignore_index=True)
trainSet= np.concatenate((np.ones( trainX.shape[0]), np.zeros(testX.shape[0]))).astype(bool)

# trainY.to_pickle( 'trainY_raw')
# combX.to_pickle('combX_raw')
# pickle.dump(trainSet, open('trainSet', 'wb'))
testID= testX['ID'].values.flatten()
# pickle.dump(testID, open( 'testID', 'wb'))
# 1. Truncate test by min and max of train data
for col in trainX.columns.values:
    l1= min( trainX[col])
    l2= max( trainX[col])
    testX.ix[ testX[col]< l1 ,col]= l1
    testX.ix[ testX[col]> l2 ,col]= l2
combX= pd.concat( (trainX, testX), axis=0, ignore_index=True)
del combX['ID']
del trainX
del testX
# 4. count missing_fill, then Replace missing_fill by the median
missing_fill= 9999999999
# count missing_fill
num_missing_fill=[]
for i in range( combX.shape[0]):
    tmp= combX.ix[i].value_counts()
    if 0 in tmp.index.values.flatten():
        num_missing_fill.append( tmp[0])
    else:
        num_missing_fill.append( 0)
combX['num_missing_fill']= num_missing_fill

### check and Replace missing_fill
for col in combX.columns.values:
    tmp= combX[col]== missing_fill
    if ( np.any( tmp)):
        a= combX[col].median(skipna= True)
        combX.ix[tmp, col]= a
        print( col, end='\t')
        print( a)


# 2. delete the constant columns
col_del= []
print('del const columns')
for col in combX.columns.values:
    if ( len( np.unique( combX[col].values.flatten()))==1):
        col_del.append( col)
        print('del '+ col)
combX.drop(labels=col_del, axis= 1, inplace=True)

# 3. del identical columns
col= combX.columns.values
col_del= set()
for i in range(len( col)):
    for j in range( i+1, len( col), 1):
        tmp1= combX[col[i]].values.flatten()
        tmp2= combX[col[j]].values.flatten()
        if np.array_equal(tmp1, tmp2):
            col_del.add( col[j])
print('del the redundant cols: ')
print( col_del)
combX.drop( labels= list(col_del), axis=1, inplace=True)


# 6. factorize col with few levels
# for col with 2 levels, treat it as dummy variable
dummy= []
for col in combX.columns.values:
    if len( np.unique( combX[col].values.flatten()))<3:
        dummy.append(True)
    else:
        dummy.append(False)
dummy= np.array(dummy)
combX_dummy= combX.ix[:, dummy]
print( 'dummy:'+ str( len( combX_dummy.columns.values.flatten())))
print( combX_dummy.columns.values.flatten())
combX_dummy_col= combX_dummy.columns.values
formula= '+'.join( combX_dummy.columns.values.flatten())
combX_dummy= patsy.dmatrix( formula, data= combX_dummy)
combX_dummy= pd.DataFrame( combX_dummy).ix[:, 1:]
combX_dummy.columns= combX_dummy_col

# bind dummy and non dummy together
combX= combX.ix[:, ~dummy]
combX= pd.concat( [combX_dummy, combX], axis=1)

# 5. count 0 for each row
num_0=[]
for i in range( combX.shape[0]):
    tmp= combX.ix[i,].value_counts()
    if 0 in tmp.index.values.flatten():
        num_0.append( tmp[0])
    else:
        num_0.append(0)
combX['num_0']= num_0

#print( combX.columns)
# 7. log transform var38
combX['log_var38']= np.log( combX['var38'].values.flatten())
#print( combX['log_var38'])
del combX['var38']


# 9. write combX in file
# combX.to_pickle('combX')

params= {'booster': 'gbtree',
    'eta': 0.01,
    'subsample': .5,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6}
    
trainX= combX.ix[trainSet,:]
testX= combX.ix[~ trainSet,]
GBM= xgb.train( params= params,
                num_boost_round=595,
                dtrain= xgb.DMatrix( trainX.values, label= trainY.values.flatten()))
test_pred= GBM.predict(data=xgb.DMatrix( testX.values))
pred_res= pd.DataFrame()
pred_res['ID']= testID
pred_res['TARGET']= test_pred
t= datetime.datetime.today()
pred_res.to_csv('pred_GBM'+ t.strftime('_%Y%m%d_%H%M%S')+'.csv' ,index=False)

