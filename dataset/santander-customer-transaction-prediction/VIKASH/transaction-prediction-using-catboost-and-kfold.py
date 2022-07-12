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

# Importing the dataset
train= pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')


train_id = train.ID_code
test_id = test.ID_code
target = train.target
train.drop(columns=["ID_code", "target"], inplace=True)
test.drop(columns=["ID_code"], inplace=True)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
test = sc_X.fit_transform(test)
from sklearn.metrics import accuracy_score  
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

def f(train,test,classifier_filename,n):
    for i in n:
        
        print(classifier_filename[1][i])
        
    
        n_split = 8
        kf = KFold(n_splits=n_split, random_state=42, shuffle=True)
    
        y_valid_pred = 0 * target
        y_test_pred = 0


        for idx, (train_index, valid_index) in enumerate(kf.split(train)):
            y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
            X_train, X_valid = train.iloc[train_index,:], train.iloc[valid_index,:]
            X_train = sc_X.transform(X_train)
            X_valid = sc_X.transform(X_valid)
            '''_train = Pool(X_train, label=y_train)
            _valid = Pool(X_valid, label=y_valid)'''
            
            print( "\nFold ", idx)
            
            #for catboost
            if(classifier_filename[1][i]=="CatBoostClassifier.csv"):

                classifier_filename[0][i].fit(X_train,y_train,eval_set=(X_valid,y_valid),use_best_model=True,verbose=500)
            else:
                classifier_filename[0][i].fit(X_train,y_train)

        
            y_pred=classifier_filename[0][i].predict(X_valid)
        
            # Making the Confusion Matrix
            cm = confusion_matrix(y_valid, y_pred)
            print("cm=%s" %cm)
        
            #accuracy_score
            score=accuracy_score(y_valid,y_pred)
            print("score=%.4g"%score)
            #roc_score
            roc=roc_auc_score(y_valid,classifier_filename[0][i].predict_proba(X_valid)[:,1])
            print("roc=%.4g"%roc)
            #classification_report
            cr=classification_report(y_valid,y_pred)
            print("cr=%s"%cr)
        
            y_test_pred +=classifier_filename[0][i].predict_proba(test)[:,1]
            
        
    
        y_test_pred=y_test_pred/n_split
        '''for i in range(0,200000):
            y_test_pred_1.append(int(y_test_pred[i]/5))'''
    
        submission=pd.DataFrame()
        submission['ID_code']=test_id
        submission['target'] = y_test_pred
        submission.to_csv(classifier_filename[1][i],index=False)

'''
classifier=[]
from sklearn.linear_model import LogisticRegression
classifier.append(LogisticRegression(solver='sag',random_state = 0,n_jobs=1))

from sklearn.neighbors import KNeighborsClassifier
classifier.append(KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2,n_jobs=-1))

from sklearn.svm import SVC
classifier.append(SVC(kernel = 'linear', random_state = 0))

from sklearn.svm import SVC
classifier.append(SVC(kernel = 'rbf', random_state = 0))

from sklearn.naive_bayes import GaussianNB
classifier.append(GaussianNB())

from sklearn.tree import DecisionTreeClassifier
classifier.append(DecisionTreeClassifier(criterion = 'entropy', random_state = 0))

from sklearn.ensemble import RandomForestClassifier
classifier.append(RandomForestClassifier())

filename=["logistic_regression.csv","kneighbour.csv","kernel_linear.csv","kernwl_rbf.csv","gaussianNB.csv","decision_tree.csv","random_forest_tree.csv"]

classifier_filename=[]
classifier_filename.append(classifier)
classifier_filename.append(filename)


f(train,test,classifier_filename)'''

##boosting model

boosting_classifier_filename=[]
boosting_classifier=[]
filename=[]
#adaboost
from sklearn.ensemble import AdaBoostClassifier #For Classification
#from sklearn.ensemble import AdaBoostRegressor #For Regression
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier() 
boosting_classifier.append(AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1.0))
filename.append("AdaBoostClassifier.csv")
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it ac# cepts sample weight 
#clf.fit(X_train,y_train)

#gradient boost
#
from sklearn.ensemble import GradientBoostingClassifier #For Classification
#from sklearn.ensemble import GradientBoostingRegressor #For Regression
boosting_classifier.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1))
#clf.fit(X_train, y_train)
filename.append("GradientBoostingClassifier.csv")
#XgBoost
#
from xgboost.sklearn import XGBClassifier
boosting_classifier.append(XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=3, min_child_weight=1, gamma=0,subsample=1,colsample_bytree=1,objective= 'binary:logistic',nthread=None,scale_pos_weight=1,seed=27))
filename.append("XGBClassifier.csv")


#lgbm
#not installed
#
'''  import lightgbm as lgb
train_data=lgb.Dataset(x_train,label=y_train)
param = {'num_leaves':150, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']
num_round=50
start=datetime.now()
lgbm=lgb.train(param,train_data,num_round)
stop=datetime.now()
#Execution time of the model
execution_time_lgbm = stop-start
execution_time_lgbm
'''
           
           

#catboost
from catboost import CatBoostClassifier
boosting_classifier.append(CatBoostClassifier(loss_function="Logloss",eval_metric="AUC",task_type="GPU",learning_rate=0.008,iterations=14000,random_seed=42,od_type="Iter",depth=10,early_stopping_rounds=500))
filename.append("CatBoostClassifier.csv")


boosting_classifier_filename.append(boosting_classifier)
boosting_classifier_filename.append(filename)

f(train,test,boosting_classifier_filename,n=[3])