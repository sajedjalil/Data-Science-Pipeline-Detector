from pandas import DataFrame,read_csv,notnull,isnull,get_dummies,concat,unique,merge
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
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
        scale=MinMaxScaler()
        scale.fit(self.df.loc[:,num_list])
        num_df=scale.transform(self.df.loc[:,num_list])
        cat_temp=get_dummies(self.df[cat_list[0]],prefix=cat_list[0])
        for i_col in range(1,len(cat_list)):
            cat_temp=concat([cat_temp,get_dummies(self.df.loc[:,cat_list[i_col]],prefix=cat_list[i_col])],axis=1)
        self.proc_df=np.concatenate([num_df,cat_temp.as_matrix()],axis=1)
        train_index=self.proc_df[:,self.proc_df.shape[1]-1]==1
        test_index=self.proc_df[:,self.proc_df.shape[1]-2]==1
        self.pca_analysis()
        self.proc_train=self.proc_df[train_index,:]
        self.proc_test=self.proc_df[test_index,:]
        pass
    def pca_analysis(self):
        pc=PCA(n_components=100)
        pc.fit(self.proc_df)
        self.proc_df=pc.transform(self.proc_df)
        #print(pc.explained_variance_ratio_)
        print('% of variance explained:',np.sum(pc.explained_variance_ratio_))
        pass
    def XGClassifier(self):
        sv=GradientBoostingRegressor()
        #params={'n_estimators':[50],'random_state':[7],'learning_rate':[0.19],'max_depth':range(1,30,5),'min_samples_split':range(20,200,40)}
        params={}
        cf=GridSearchCV(sv,params,scoring='r2',verbose=3,cv=5)
        cf.fit(self.proc_train,self.Y_train['y'])
        print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
        self.xg_predictions=cf.predict(self.proc_test)
        self.xg_prediction_train=cf.predict(self.proc_train)
        print('Mean Squared Error:')
        print(mean_squared_error(self.Y_train['y'],self.xg_prediction_train))
        print('R square:')
        print(r2_score(self.Y_train['y'],self.xg_prediction_train))
        self.xg=cf
mc=Merc()
mc.load_data()
mc.preprocess_data()
mc.XGClassifier()
columns = ['y']
sub = DataFrame(data=np.exp(mc.xg_predictions), columns=columns)
sub['ID'] = mc.test['ID']
sub = sub[['ID','y']]
sub.to_csv("xgb_tuned_v1.csv", index=False)