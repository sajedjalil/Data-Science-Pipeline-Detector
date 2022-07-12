# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Any results you write to the current directory are saved as output.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import patsy
import xgboost as xgb
import sklearn as sk
import pickle

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

# 2. delete the constant columns
combX= pd.concat( (trainX, testX), axis=0, ignore_index=True)
col_del= []
print('del const columns')
for col in combX.columns.values:
    if ( len( np.unique( combX[col].values.flatten()))==1):
        col_del.append( col)
        print('del '+ col)
trainX.drop(labels=col_del, axis= 1, inplace=True)
testX.drop( labels= col_del, axis=1, inplace=True)

# 3. del identical columns
combX= pd.concat( (trainX, testX), axis=0, ignore_index=True)
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
trainX= combX.ix[trainSet]
testX= combX.ix[~ trainSet]

# 4. Partition predictors into 2 parts, factors (discrete) F, and continuous C
level_lim= 2
F_col=set()
C_col=set( combX.columns.values)
for col in combX.columns.values:
    if len( np.unique( combX[col]))<=level_lim:
        F_col.add(col)
        C_col.discard(col)

combX_F= combX[list( F_col)]
combX_C= combX[list( C_col)]


# trainX_F= combX_F.ix[ trainSet,]
# trainX_C= combX_C.ix[ trainSet,]
# testX_F= combX_F.ix[ ~ trainSet,]
# testX_C= combX_C.ix[ ~ trainSet,]
# trainX_F.to_pickle('raw_trainX_F')
# trainX_C.to_pickle('raw_trainX_C')
# testX_F.to_pickle('raw_testX_F')
# testX_C.to_pickle('raw_testX_C')
# combX.to_pickle('tmp_combX')

# count number of 0 in each obs
num_0= []
for i in range(  combX.shape[0]):
    tmp= combX.ix[i].values.flatten()
    num_0.append( np.sum([1 if x==0 else 0 for x in tmp]))
del combX
del trainX
del testX

# 5. In continuous data,
# delta_imp_aport_var13_1y3 and
# delta_imp_compra_var44_1y3
# have filled missing values (as 9999999999)
# we fill it by the median (0)
missing_fill= 9999999999
### check missing_fill
# for col in combX_C.columns.values:
#     tmp= combX_C[col]== missing_fill
#     if ( np.any( tmp)):
#         print( col)
###

combX_C.ix[ combX_C['delta_imp_aport_var13_1y3']==missing_fill, 'delta_imp_aport_var13_1y3']=0
combX_C.ix[ combX_C['delta_imp_compra_var44_1y3']==missing_fill, 'delta_imp_compra_var44_1y3']=0

# 6. add num_0 to combX_C
combX_C['num_0']= num_0

# 7. log transform var38
combX_C['log_var38']= np.log( combX_C['var38'].values.flatten())
del combX_C['var38']

# 8. Generate the one-hot encoding matrix from combX_F
# oneHot_df=[]
# for col in combX_F.columns.values:
#     formula= 'C( '+ col+ ','+ 'Treatment)'
#     t1= patsy.dmatrix(formula, data= combX_F)
#     t1= pd.DataFrame( t1, columns=t1.design_info.column_names)
#     del t1['Intercept']
#     oneHot_df.append( t1)
# combX_F= pd.concat( oneHot_df, axis= 1, ignore_index=True)
formula= '+'.join( combX_F.columns.values.flatten())
combX_F= patsy.dmatrix( formula, data= combX_F)
combX_F= pd.DataFrame( combX_F, columns=combX_F.design_info.column_names)
del combX_F['Intercept']



# 9. write combX_F and combX_C in file
# combX_F.to_pickle('combX_F_p')
# combX_C.to_pickle('combX_C_p')
combX= pd.concat( [combX_F, combX_C], ignore_index=True, axis=1)
trainX= combX.ix[trainSet]
testX= combX.ix[~ trainSet]


params= {'booster': 'gbtree',
    'eta': 0.01,
    'subsample': .5,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3}
## Cross Validation to choose max_depth and n_estimators
tmp= {}

for depth in range( 5, 6,1):
    params['max_depth']= depth
    CV= xgb.cv( params= params,verbose_eval= False, 
        num_boost_round=600,
        dtrain= xgb.DMatrix(trainX.values, label= trainY.values.flatten()),
        nfold= 5)
    best_score= np.max(CV.ix[:, 0])
    best_n_est= np.argmax( CV.ix[:,0])
    tmp[depth]= (best_n_est, best_score, CV)
    print( depth , end='\t')
    print( best_score, end='\t')
    print( best_n_est)

pickle.dump( tmp, open( 'GBM_trainFeature_CV', 'wb'))
for key, values in tmp.items():
    print(key, end='\t')
    print( values[0], end='\t')
    print( values[1])

