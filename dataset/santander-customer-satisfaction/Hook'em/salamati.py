#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

#from __future__ import division

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
#import xgboost as xgb
#from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
#from sklearn.pipeline import Pipeline
#from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler #Scaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "D:/temp/santander"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns

for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)


y_train = df_train['TARGET']
X_train = df_train.drop(['ID','TARGET'], axis=1)
#X_train = X_train[X_train.column.isin(feature_names)]

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1)
#X_test = X_test[X_test.column.isin(feature_names)]

len_train = len(X_train)
len_test  = len(X_test)

model = MLPClassifier(hidden_layer_sizes=(20,20), activation='relu', algorithm='sgd',
     beta_1=0.9, beta_2=0.999,momentum=0.9,nesterovs_momentum=True,alpha=1e-5,
     learning_rate_init=0.001, max_iter = 1000, random_state = 720, 
     learning_rate='adaptive')

mean_auc = 0.0
n = 2  # number of folds in strattified cv
kfolder=StratifiedKFold(y_train, n_folds= n,shuffle=True, random_state=51)     
i=0
for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
    # creaning and validation sets
    X_tr, X_cv = X_train.iloc[list(train_index)], X_train.iloc[list(test_index)]
    y_tr, y_cv = np.array(y_train)[train_index], np.array(y_train)[test_index]
    # do scalling
    scaler=StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_cv = scaler.transform(X_cv)

    # train model
    model.fit(X_tr,y_tr)
    #  make predictions in probabilities
    preds=model.predict_proba(X_cv)[:,1]

    # compute AUC metric for this CV fold
    roc_auc = roc_auc_score(y_cv, preds)
    print ("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
    mean_auc += roc_auc
    i+=1
        
mean_auc/=n
print (" Average AUC: %s" % str(mean_auc) ) 
