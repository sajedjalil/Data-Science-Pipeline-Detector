from sklearn.svm import SVR
from pandas import DataFrame,read_csv,notnull,isnull,get_dummies,concat,unique
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,log_loss,mean_squared_error,r2_score
seed=7
np.random.seed(seed)
class Merc:
    def load_data(self):
        self.train=read_csv('../input/train.csv',header='infer')
        self.test=read_csv('../input/test.csv',header='infer')
        self.train.loc[:,'y']=np.log(self.train.loc[:,'y'])
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
        scale=StandardScaler()
        scale.fit(self.df.loc[:,num_list])
        num_df=scale.transform(self.df.loc[:,num_list])
        cat_temp=get_dummies(self.df[cat_list[0]],prefix=cat_list[0])
        for i_col in range(1,len(cat_list)):
            cat_temp=concat([cat_temp,get_dummies(self.df.loc[:,cat_list[i_col]],prefix=cat_list[i_col])],axis=1)
        self.proc_df=np.concatenate([num_df,cat_temp.as_matrix()],axis=1)
        vt=VarianceThreshold(threshold=0.001)
        self.proc_df=vt.fit_transform(self.proc_df)
        self.proc_train=self.proc_df[self.proc_df[:,self.proc_df.shape[1]-1]==1,:self.proc_df.shape[1]-2]
        self.proc_test=self.proc_df[self.proc_df[:,self.proc_df.shape[1]-2]==1,:self.proc_df.shape[1]-2]
        pass
    def Gradboosting(self):
        grad=xgb.XGBRegressor()
        params={'booster':['gbtree'],'gamma':[0.01],'learning_rate':[0.1],'max_depth':[3],'n_estimators':[500],'n_jobs':[-1],'random_state':[seed],'subsample':[1]}
        cf=GridSearchCV(grad,param_grid=params,cv=3,scoring='r2',verbose=2)
        cf.fit(self.proc_train,self.Y_train['y'])
        print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
        self.grad_predictions=cf.predict(self.proc_test)
        self.gradtree=cf
    def stack_learners(self,learners,stacker,scoring):
        C={}
        D={}
        for i in learners:
            cf=GridSearchCV(learners[i]['Model'],param_grid=learners[i]['Params'],scoring=scoring,verbose=3,cv=2)
            cf.fit(self.proc_train,self.Y_train['y'])
            C[i]=cf.predict(self.proc_train)
            D[i]=cf.predict(self.proc_test)
        self.new_train=DataFrame(C)
        self.new_test=DataFrame(D)
        #print(self.new_train.head())
        print(stacker)
#        scale=MinMaxScaler()
#        scale.fit(np.append(self.new_train,self.new_test,axis=0))
#        self.new_train=scale.transform(self.new_train)
#        self.new_test=scale.transform(self.new_test)
        #cf=GridSearchCV(stacker['Model'],param_grid=stacker['Params'],scoring='r2',verbose=3)
        #cf.fit(self.new_train,self.Y_train['y'])
        #print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
        self.stack_predictions=self.new_test
        print (self.new_test.columns)
        #cf.predict(self.new_test)
mc=Merc()
mc.load_data()
mc.preprocess_data()
#mc.Gradboosting()
mc.stack_learners({'Random Forest':{'Model':RandomForestRegressor(),'Params':{'n_jobs': [-1], 'n_estimators': [1000], 'random_state': [7], 'min_samples_split': [8]}},'GradientBoost':{'Model':xgb.XGBRegressor(),'Params':{'booster':['gbtree'],'gamma':[0.01],'learning_rate':[0.1],'max_depth':[3],'n_estimators':[500],'n_jobs':[-1],'random_state':[seed],'subsample':[1]}}},'Bagging','r2')
final=DataFrame(mc.test['ID'])
a=np.exp(mc.stack_predictions.iloc[:,0])
b=np.exp(mc.stack_predictions.iloc[:,1])
final['y']=(3*a+2*b)/5
final.to_csv('xgb_out.csv',index=False)
#xg=xgb.XGBRegressor()
#print(xg.get_params().keys())
