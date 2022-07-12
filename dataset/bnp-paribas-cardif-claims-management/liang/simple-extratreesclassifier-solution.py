# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Class to predict the probability
class Class_Predict_BNP:

 def Predict_BNP(self,df_train,df_test,target):
    # Prepraing Train Test Set        
    y = target
    X = df_train.values

    # Select Model
    clf = ExtraTreesClassifier(n_estimators=750,max_features=50, \
                               criterion= 'entropy',min_samples_split= 4, \
                               max_depth= 35, min_samples_leaf= 2, \
                               n_jobs = -1, random_state=12)
                               
    clf.fit(X,y)

    # Prediction probability
    value_proba=clf.predict_proba(df_test.values)
    
    # Submission
    submission=pd.read_csv('../input/sample_submission.csv')
    submission.index=submission.ID
    submission.PredictedProb=value_proba[:,1]
    submission.to_csv('./BNP_ETC_Proba.csv', index=False)
    return;

# Load Data
df_train = pd.read_csv('../input/train.csv')     # 114321 rows x 133 columns
df_test = pd.read_csv('../input/test.csv')  # 114393 rows x 132 columns

# Drop columns
target = df_train['target'].values
df_train=df_train.drop(['ID','target'],axis=1)
df_test=df_test.drop(['ID'],axis=1)

# Feature Processing
refcols=df_train.columns
df_train=df_train.fillna(-999)
df_test=df_test.fillna(-999)

for elt in refcols:
    if df_train[elt].dtype=='O':
        df_train[elt], temp = pd.factorize(df_train[elt])
        df_test[elt]=temp.get_indexer(df_test[elt])
    else:
        df_train[elt]=df_train[elt].round(5)
        df_test[elt]=df_test[elt].round(5)

# Call the class and funtion to get prediction probability file.        
a = Class_Predict_BNP()
a.Predict_BNP(df_train,df_test,target)