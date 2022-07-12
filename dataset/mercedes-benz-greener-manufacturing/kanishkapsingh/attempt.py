# -*- coding: utf-8 -*-
"""
Created on Sat Jun 03 04:01:35 2017

@author: Kanishka105457
"""

#%%
from sklearn.svm import SVR
from pandas import DataFrame,read_csv,notnull,isnull,get_dummies,concat,unique
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,log_loss,mean_squared_error,r2_score
#%%
seed=7
np.random.seed(seed)
class Merc:
    def load_data(self):
        self.train=read_csv('../input/train.csv',header='infer')
        self.test=read_csv('../input/test.csv',header='infer')
        self.train.loc[:,'y']=self.train.loc[:,'y']
        self.train_corr=self.train.corr()
        self.Y_train=self.train[['ID','y']]
        del self.train['y']
        self.train['Source']='train'
        self.test['Source']='test'
        self.df=self.train.append(self.test)
        pass
    def preprocess_data(self):
        cat_list=list(self.df.columns[self.df.dtypes=='object'])
        num_list=list(self.df.columns[self.df.dtypes!='object'])
        num_list.pop(0)
        #scale=StandardScaler()
        #scale.fit(self.df.loc[:,num_list])
        num_df=self.df.loc[:,num_list]
        cat_temp=get_dummies(self.df[cat_list[0]],prefix=cat_list[0])
        for i_col in range(1,len(cat_list)):
            cat_temp=concat([cat_temp,get_dummies(self.df.loc[:,cat_list[i_col]],prefix=cat_list[i_col])],axis=1)
        self.proc_df=np.concatenate([num_df,cat_temp.as_matrix()],axis=1)
        #vt=VarianceThreshold(threshold=0.01)
        #self.proc_df=vt.fit_transform(self.proc_df)
        self.proc_train=self.proc_df[self.proc_df[:,self.proc_df.shape[1]-1]==1,:self.proc_df.shape[1]-2]
        self.proc_test=self.proc_df[self.proc_df[:,self.proc_df.shape[1]-2]==1,:self.proc_df.shape[1]-2]
        pass
    def SVClassifier(self):
        sv=SVR()
        params={'kernel':['rbf'],'C':[0.2]}
        cf=GridSearchCV(sv,params,scoring='r2',verbose=2,cv=10)
        cf.fit(self.proc_train,self.Y_train['y'])
        print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
        self.sv_predictions=cf.predict(self.proc_test)
        #sv_accuracy=accuracy_score(self.y_test,sv_predictions)
        #sv_predictions_prob=cf.predict_proba(self.test)
        #sv_log_loss=log_loss(self.y_test,sv_predictions_prob)
        #print('SVC created with an accuracy of ',sv_accuracy,' and a log loss of ',sv_log_loss)
        self.svc=cf
    def Gradboosting(self):
        grad=GradientBoostingRegressor()
        params={'loss':['ls'],'learning_rate':[0.1],'n_estimators':[1000],'max_depth':[2],'max_features':['auto'],'random_state':[seed]}
        cf=GridSearchCV(grad,param_grid=params,cv=5,scoring='r2',verbose=2)
        cf.fit(self.proc_train,self.Y_train['y'])
        print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
        self.grad_predictions=cf.predict(self.proc_test)
        #grad_predictions_prob=cf.predict_proba(self.test)
        #grad_accuracy=accuracy_score(self.y_test,grad_predictions)
        #grad_log_loss=log_loss(self.y_test,grad_predictions_prob)
        #print('Decision Tree created with an accuracy of ',grad_accuracy,' and a log loss of ',grad_log_loss)
        self.gradtree=cf
#%%
mc=Merc()
mc.load_data()
mc.preprocess_data()
#%%
#mc.SVClassifier()
#%%
mc.Gradboosting()
#%%
final=DataFrame(mc.test['ID'])
final['y']=mc.grad_predictions
final.to_csv('sv_pred.csv',index=False)