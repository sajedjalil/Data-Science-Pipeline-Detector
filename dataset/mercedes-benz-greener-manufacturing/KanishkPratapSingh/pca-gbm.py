from pandas import DataFrame,read_csv,notnull,isnull,get_dummies,concat,unique,merge
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score,log_loss,mean_squared_error,r2_score
from sklearn.metrics.scorer import make_scorer
seed=7
np.random.seed(seed)
class Merc:
    def load_data(self):
        self.train_csv=read_csv('../input/train.csv',header='infer')
        self.train,self.test,self.Y_train,self.Y_test=train_test_split(self.train_csv.drop('y',axis=1),self.train_csv['y'],test_size=0.2,random_state=seed)
        self.sub=read_csv('../input/test.csv',header='infer')
        self.Y_train=np.log(self.Y_train)
        self.train_corr=self.train.corr()
        #self.Y_train=self.train[['ID','y']]
        #del self.train['y']
        self.train['Source']='train'
        self.test['Source']='test'
        self.sub['Source']='sub'
        self.df=self.train.append(self.test)
        self.df=self.df.append(self.sub)
        print('Train:',self.train.shape,' Test:',self.test.shape,' DF:',self.df.shape,'Sub',self.sub.shape)
        pass
    def preprocess_data(self):
        cat_list=list(self.df.columns[self.df.dtypes=='object'])
        num_list=list(self.df.columns[self.df.dtypes!='object'])
        num_list.pop(0)
        scale=MinMaxScaler()
        scale.fit(self.df.loc[:,num_list])
        num_df=scale.transform(self.df.loc[:,num_list])
        cat_temp=get_dummies(self.df[cat_list[0]],prefix=cat_list[0])
        for i_col in range(1,len(cat_list)):
            cat_temp=concat([cat_temp,get_dummies(self.df.loc[:,cat_list[i_col]],prefix=cat_list[i_col])],axis=1)
        self.proc_df=np.concatenate([num_df,cat_temp.as_matrix()],axis=1)
        train_index=self.proc_df[:,self.proc_df.shape[1]-1]==1
        test_index=self.proc_df[:,self.proc_df.shape[1]-2]==1
        sub_index=self.proc_df[:,self.proc_df.shape[1]-3]==1
        self.pca_analysis()
        self.proc_train=self.proc_df[train_index,self.proc_df.shape[1]-3]
        self.proc_test=self.proc_df[test_index,self.proc_df.shape[1]-3]
        self.proc_sub=self.proc_df[sub_index,self.proc_df.shape[1]-3]
        print('Proc Train:',self.proc_train.shape,' Proc Test:',self.proc_test.shape,' Proc DF:',self.proc_df.shape,'Proc Sub',self.proc_sub.shape)
        pass
    def pca_analysis(self):
        pc=PCA()
        pc.fit(self.proc_df)
        self.proc_pca=pc.transform(self.proc_df)
        self.proc_df=np.concatenate([self.proc_df,self.proc_pca],axis=1)
        #print(pc.explained_variance_ratio_)
        print('% of variance explained:',np.sum(pc.explained_variance_ratio_),self.proc_pca.shape)
        pass
    def cus_scorer(self,true_pred,pred):
        return mean_squared_error(true_pred,pred)
    def rec_call(self,call_num):
        for j in self.params[self.length-1]:
            if self.length-1==0:
                #Classify code
                print('I am classifying using ',self.params[self.length-1])
            else:
                self.length=self.length-1
                self.rec_call(self.length)
    def call_recurser(self,params):
        self.params=list(params.keys())
        self.length=len(self.params)
        self.rec_call(self.length)
    def XGClassifier(self,params):
        self.scorer={'Params':[],'R_2 Scores':[],'MSE Scores':[]}
        for i in params['learning_rate']:
            for j in params['n_estimators']:
                ind={'subsample':1,'learning_rate':i,'n_estimators':j,'min_samples_split':140,'max_depth':4,'min_samples_leaf':2}
                print('Fitting using the following:',ind)
                cf=GradientBoostingRegressor(**ind)
                #self.my_scorer=make_scorer(self.cus_scorer,greater_is_better=False)
                #params={'n_estimators':[70],'learning_rate':[0.05],'max_depth':[3],'min_samples_split':[400],'min_samples_leaf':range(5,100,20)}
                #cf=GridSearchCV(sv,params,scoring=self.my_scorer,verbose=3,cv=5)
                cf.fit(self.proc_train,self.Y_train)
                #print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
                self.xg_predictions=np.exp(cf.predict(self.proc_train))
                self.xg_prediction_test=np.exp(cf.predict(self.proc_test))
                print('Mean Squared Error Train',mean_squared_error(np.exp(self.Y_train),self.xg_predictions),' R square Train:',r2_score(np.exp(self.Y_train),self.xg_predictions))
                print('Mean Squared Error Test',mean_squared_error(self.Y_test,self.xg_prediction_test),' R square Test:',r2_score(self.Y_test,self.xg_prediction_test))
                self.scorer['Params'].append(ind)
                self.scorer['R_2 Scores'].append(r2_score(self.Y_test,self.xg_prediction_test))
                self.scorer['MSE Scores'].append(mean_squared_error(self.Y_test,self.xg_prediction_test))
        print('Max R_2 is ',np.max(self.scorer['R_2 Scores']),' using ',self.scorer['Params'][np.argmax(self.scorer['R_2 Scores'])])
        self.xg=cf
mc=Merc()
mc.load_data()
mc.preprocess_data()
#mc.XGClassifier({'learning_rate':[0.01,0.02,0.05,0.1],'n_estimators':[50,150,250,350,450,550]})
#mc.call_recurser({'n_estimators':[40,50,70],'learning_rate':[0.1,0.2]})
#columns = ['y']
#sub = DataFrame(data=np.exp(mc.xg_predictions), columns=columns)
#sub['ID'] = mc.test['ID']
#sub = sub[['ID','y']]
#sub.to_csv("xgb_tuned_v1.csv", index=False)