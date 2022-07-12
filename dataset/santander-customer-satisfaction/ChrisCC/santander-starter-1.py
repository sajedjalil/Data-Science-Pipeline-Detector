# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline, metrics, grid_search, cross_validation
import pandas as pd
import numpy as np 
import scipy as sc
from sklearn import preprocessing, feature_extraction, ensemble, pipeline, metrics, grid_search, pipeline, metrics, grid_search
import time
from datetime import datetime
from scipy.stats import rankdata
import math
import random
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,ElasticNet
from sklearn.grid_search import ParameterGrid
# Load data
start = time.time()    
#Load data
trainData = pd.read_csv('../input/train.csv')
print ("Loading finished in %0.3fs" % (time.time() - start))        
testData = pd.read_csv('../input/test.csv')
print ("Loading finished in %0.3fs" % (time.time() - start))   

trainSize=trainData.shape[0]
testSize=testData.shape[0]
fullData=pd.concat([trainData,testData])

# zero variance
fullVar = pd.DataFrame.var(fullData)
zeroVarCols = fullVar[fullVar==0].index.values
print ("There are %d columns with zero variances: %s " %(zeroVarCols.size, zeroVarCols))

dupCols = ['ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41'
           , 'ind_var46_0', 'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27'
           , 'num_var41', 'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41'
           , 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3'
           , 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1'
           , 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3'
           , 'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3'
           , 'ind_var29_0', 'ind_var29', 'ind_var13_medio', 'ind_var18', 'ind_var26', 'ind_var25'
           , 'ind_var32', 'ind_var34', 'ind_var37', 'ind_var39', 'num_var29_0', 'num_var29', 'num_var13_medio'
           , 'num_var18', 'num_var26', 'num_var25', 'num_var32', 'num_var34', 'num_var37', 'num_var39', 'saldo_var29'
           , 'saldo_medio_var13_medio_ult1', 'delta_num_reemb_var13_1y3', 'delta_num_reemb_var17_1y3', 'delta_num_reemb_var33_1y3'
           , 'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_in_1y3'
           , 'delta_num_trasp_var33_out_1y3']

trainCols = list(trainData.columns.values)
print (len(trainCols))

trainCols.remove('ID')
trainCols.remove('TARGET')

print (len(trainCols))

for c in list(set(dupCols)|set(zeroVarCols)):
    trainCols.remove(c)    
print (len(trainCols))

print (len(trainCols))

start = time.time()    
model = xgb.XGBClassifier(max_depth=5
                  , learning_rate=0.02
                  , n_estimators=560
                  , silent=False
                  , objective="binary:logistic"
                  , subsample=0.7
                  , colsample_bytree=0.7
                  , min_child_weight = 1
                  , seed = 1234
                  , nthread = -1)

model.fit(fullData[:trainSize][trainCols].fillna(-999).values, fullData[:trainSize]['TARGET'].fillna(-999).values)

print ("Training finished in %0.3fs" % (time.time() - start))     
predictYDf = pd.DataFrame(model.predict_proba (fullData[trainSize:][trainCols].fillna(-999).values)[:,1])

submission = pd.DataFrame({"ID": fullData['ID'][trainSize:].values.reshape(testSize),"TARGET": predictYDf.values.reshape(testSize)})
submission.to_csv('submission_mean_var15.csv',index=False)  


print ("Predicting finished in %0.3fs" % (time.time() - start))     

print ("Submission created.")

