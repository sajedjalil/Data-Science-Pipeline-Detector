import sklearn.linear_model as lm 
import sklearn.tree as skt
import sklearn.ensemble as ske
import sklearn.metrics as skm
import sklearn.svm as ssvm
from sklearn import linear_model
import sklearn.model_selection as skms
import sklearn.externals as ext

from scipy.stats import randint as sp_randint
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import pandas.plotting as pplot

import os
print(os.listdir("../input"))


def load_banking_data(path):
    return pd.read_csv(path)

def split_data_set(path):
    dframe = load_banking_data(path)
    X = dframe.drop('target',axis=1)
    X = X.drop('ID',axis=1)
    y = dframe['target'].copy()
    train_set,test_set,train_labels,test_labels = ms.train_test_split(X,y,test_size=0.1,random_state=42)
    return [train_set,test_set,train_labels,test_labels]

class DataFrameToValues(BaseEstimator,TransformerMixin):
    def __init__(self,attributes):
        self.attributes = attributes
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        arr = X[self.attributes].values
        return arr

def construct_pipeline(num_attribs):
    return pline.Pipeline([
        ('dftovalue',DataFrameToValues(num_attribs)),
        ('imputer',skp.Imputer(strategy='median')),
        ('std_scaler',skp.StandardScaler())
    ])

def train_randomforest(train_set,train_labels,attr_params):
    data_pipeline = satanderpipeline.construct_pipeline(attr_params)
    feed_data = data_pipeline.fit_transform(train_set)
    param_grid = {
        'max_features':sp_randint(1,28),
        'max_depth': [3,None],
        'min_samples_split':sp_randint(2,28),
        'min_samples_leaf':sp_randint(1,28),
        'bootstrap':[True,False],
        #'criterion':['gini','entropy']
    }

    rf = ske.RandomForestRegressor(n_estimators=20)
    #return rf.fit(data_pipeline.fit_transform(train_set),train_labels)
    
    #grid_search = skms.GridSearchCV(rf,param_grid,cv=5,scoring='neg_mean_squared_error')
    #grid_search.fit(data_pipeline.fit_transform(train_set),train_labels)
    #return grid_search.best_estimator_
    
    random_search = skms.RandomizedSearchCV(rf,param_grid,n_iter=20,scoring='neg_mean_squared_log_error',cv=5)
    random_search.fit(feed_data,train_labels)
    return random_search.best_estimator_

def train_svmsvr(train_set,train_labels,attr_params):
    data_pipeline = satanderpipeline.construct_pipeline(attr_params)
    svmsvr = ssvm.SVR()
    svmsvr.fit(data_pipeline.fit_transform(train_set),train_labels)
    return svmsvr


def find_meanerror(model,test_data,test_labels):
    return np.sqrt(skm.mean_squared_log_error(test_labels,model.predict(test_data)))

def write_csv(id,target,filename='sample_submission.csv'):
    df = pd.DataFrame({'ID':id,'target':target})
    df.to_csv(filename,index=False)

def save_model(model,filename='model.dump'):
    ext.joblib.dump(model,filename)

def load_model(filename = 'model.dump'):
    return ext.joblib.load(filename)

if __name__ == '__main__':
    train_set,test_set,train_labels,test_labels = loaddata.split_data_set('../input/train.csv')
    dframe = loaddata.load_banking_data('../input/test.csv')
    attr_params = list(train_set)
    
    #attr_params = ['555f18bd3','9fd594eec','5bc7ab64f','cbbc9c431','f190486d6',
    #'6b119d8ce','f74e8f13d','ac30af84a','26fc93eb7','58e2e02e6','429687d5a','e8d9394a0','6eef030c1','f3cf9341c','e4159c59e',
    #'ba4ceabc5','51707c671','1702b5bf0','38e6f8d32','f296082ec','41bc25fef','f1851d155','70feb1494','0d5215715','6d2ece683',
    #'ad207f7bb','174edf08a','1fd0a1f2a','d79736965']

    #get the test data
    test_pipeline = satanderpipeline.construct_pipeline(attr_params)
    test_data = test_pipeline.fit_transform(test_set)
 

    #train the model
    rf_model = train_randomforest(train_set,train_labels,attr_params)
    svm_model = train_svmsvr(train_set,train_labels,attr_params)
    save_model(svm_model,'svm_model.dump')
    saved_model = load_model('svm_model.dump')
    #calculate the error
    
    print('Mean Error svm:',find_meanerror(svm_model,test_data,test_labels))
    #write the file for submission
    real_test_data = test_pipeline.fit_transform(dframe.drop('ID',axis=1))
    id_val = dframe['ID'].values
    predict_target = saved_model.predict(real_test_data)
    write_csv(id_val,predict_target,'svm_results.csv')
