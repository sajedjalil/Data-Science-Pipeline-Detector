import kagglegym

import numpy as np
import pandas as pd
import time
start = time.time()
import copy              
from sklearn.ensemble import (RandomForestRegressor,ExtraTreesRegressor ,
                                  BaggingRegressor, GradientBoostingRegressor  ,
                                 IsolationForest )
from sklearn.tree import DecisionTreeRegressor

from sklearn import preprocessing as prep
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import r2_score
from sklearn import grid_search
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, 
                                      HuberRegressor,BayesianRidge,LassoLars,Ridge,Lasso,ElasticNet,
                                      ARDRegression,LogisticRegression,LassoCV)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor 
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
#The "environment" is our interface for code competitions

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()
#DotProduct(sigma_0=rn, sigma_0_bounds=(1e-05, 10000))
                                      
estimators = [('RANSACReg', RANSACRegressor(base_estimator=LinearRegression(fit_intercept=False))),
        ('LinReg', LinearRegression(fit_intercept=False)),('Theil_Sen', TheilSenRegressor(fit_intercept=False)),
                  ('Ridge', Ridge(fit_intercept=False)),('HuberRegressor', HuberRegressor(fit_intercept=False)),
                 ('BayesRidge', BayesianRidge(fit_intercept=False)),('LassoLars', LassoLars(fit_intercept=False,alpha = 25)),
                  ('Lasso', Lasso(fit_intercept=False,alpha = 25)),('ElasticNet',ElasticNet(alpha=13,fit_intercept=False)),
                  ('ARDRegression',ARDRegression(fit_intercept=False))]
param_grid = {'n_estimators': np.random.randint(10,300,10),
                          'max_depth':np.random.randint(3,30,5),
                          'min_samples_split':(2,4,0.05,0.001,0.0001),
                          'max_features':('auto',0.5,0.05,1,'sqrt','log2'),
                          'min_weight_fraction_leaf':np.random.uniform(0,0.5,5)
                         }
# Note that the first observation we get has a "train" dataframe
trains=observation.train
print("Train has {} rows".format(len(trains)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))
param_grids = {'learning_rate': np.random.uniform(0.00001,0.9,10)} #'loss':('ls','lad','huber'),
                              #'max_features':('auto',0.5,0.005,10,'sqrt','log2'),
                              #'n_estimators': np.random.randint(100,200,3),
                              #,'max_depth':np.random.randint(2,50,5) , 
#print(model0.get_params().keys())
#mdl=MLPRegressor(warm_start=True)
mdl =Ridge()
#mdl=BaggingRegressor(base_estimator=ExtraTreesRegressor(max_depth=10),warm_start=True, n_jobs=-1)

mdl= BaggingRegressor(GradientBoostingRegressor(learning_rate=0.00000000000022227))
mdl=GradientBoostingRegressor(learning_rate=0.00000000000022227)
mdl=BaggingRegressor(GaussianProcessRegressor(n_restarts_optimizer=6),warm_start=True)#,warm_start=True)

#mdl=GaussianProcessRegressor(n_restarts_optimizer=36)
mdl=BaggingRegressor(base_estimator=SVR(kernel='linear', C=1e-3),warm_start=True)
seed=19781980
#good condition
#BaggingRegressor(base_estimator=SVR(kernel='linear', C=1e-4))  Timestamp you rare here 0.115240906412 False 1543
np.random.seed(seed)
#model0= RandomForestRegressor();
#', 'tol':np.random.uniform(0.0001,0.09,5),
 #'solver' :('auto', 'svd',  'lsqr', 'sparse_cg', 'sag')} #'cholesky'
print("Train has {} rows".format(len(observation.train)))

#The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))
#trains["timestamp"].unique())
#
def check_nan(dataframe,reject='no'):
    allcol = dataframe.columns;maxx=len(dataframe);bfor = maxx
    #print(dataframe.head())
    for na in allcol:
        nan0 = pd.isnull(dataframe[na]).sum()
        #print('sum of nan0',nan0,'maxx',maxx )
        
 
        if nan0/maxx >=0.70: #  drop columns with more than 70% of NaN
            #print('column removed',na)
            
            dataframe = dataframe.drop(na,1)
        else:
            if bfor>=nan0 and na!='y' and na!='id' and na!='timestamp':
                bfor=nan0;na0 = na # identified the column with minimum NaN
                #print('column kept nan',bfor)

    dfdropped = dataframe.dropna(0,subset=[na0])
    #print('dfdropped',dfdropped.columns)
    #print('dfdrop.shape:',dfdrop.shape,'dfdrop.columns',dfdrop.columns)
    return(dfdropped.fillna(dfdropped.median()))


       
#col_abov02 = ['id', 'timestamp', 'technical_2', 'technical_6', 'technical_11',
       #'technical_17', 'technical_22', 'technical_34', 'y']      
#target_var = 
## Check_nan

def PreprocessBox(dataframe,fil_method,testtrain='ok'):
    #res = dfRangeBox(pre_df[1],opt =1)
    import numpy as np
    from sklearn import preprocessing as prep
    #indx =trains["timestamp"].unique()

    #df_nona =dataframe.fillna(0) if fil_method==0 else dataframe
    #print(df_nona)
    df_nona0 = check_nan(dataframe,reject='yes')

    #df_nona = (dataframe.fillna(dataframe.mean()) if fil_method==3 else df_nona)
    df_nona1 = df_nona0.interpolate(method='akima') #if fil_method==4 else df_nona)
    df_nona = df_nona1.fillna(df_nona1.median()) #if fil_method==4 else df_nona1
    

    df_nona1 = pd.DataFrame(data=prep.robust_scale(df_nona,axis=0,quantile_range=(0,100))
                            ,columns=(df_nona.columns))
    df_nona0 = check_nan(df_nona1,reject='yes')

    #df_nona1 = pd.DataFrame(data=prep.StandardScaler(df_nona0,axis=0,quantile_range=(0,100))
                            #,columns=(df_nona0.columns))

                      
                           
  

    return(df_nona0)
#timestamp = observation.features["timestamp"][0]
    
#actual_y = list(df[df["timestamp"] == timestamp]["y"].values)
#print(len(actual_y))
#print(df[df.timestamp==906].head())
##print(observation.features.head())
#print(observation.train.head())
#print(observation.test.head())
lower_cut = -0.0860941; upper_cut = 0.0934978

rsqs =-10000000
scaler = StandardScaler()
lp=np.arange(402,452,1)
it=0;ite=0
bond=1000;ites1=0
rggg = np.arange(0,905,10);lop=True;lop1=True
tmsp = trains['timestamp']
trains = trains.drop('timestamp',1)
trains0 = PreprocessBox(trains,4,testtrain='ok')
allcol = trains0.columns
col_fund = allcol[7:70];col_der=allcol[2:7];col_tech=allcol[70:110]
y=trains0.loc[trains0.index,'y']
Xtrain =trains0.loc[trains0.index,trains0.columns.drop(col_der,1)]
Xtrain = Xtrain.loc[Xtrain.index,Xtrain.columns.drop(col_fund,1)]
todelete=['id','technical_1','technical_2','technical_3','technical_5','technical_6','technical_7','technical_9',
            'technical_10', 'technical_11', 'technical_12','technical_13',  'technical_16', 'technical_17',
            'technical_18', 'technical_19', 'technical_20', 'technical_21', 'technical_24', 'technical_25',
            'technical_27','technical_28', 'technical_30', 'technical_31','technical_32', 'technical_33', 
             'technical_35' , 'technical_36', 'technical_37', 'technical_38', 'technical_39', 'technical_40',
            'technical_41', 'technical_42', 'technical_44']
           
Xtrain = Xtrain.loc[Xtrain.index,Xtrain.columns.drop(todelete,1)]
Xtrain['timestamp']=tmsp
cols = Xtrain.columns.drop('y',1)

print(cols)
colss=[]

#model0.fit(Xtr,y)
#prd = mdl.predict(tX_test0.reshape(-1, 1))
#rrr2 =np.round(r2_score(ty_test0,prd),8)
lo_lim = 1000000;j=0;mdl_list=[]
j=j+1
while lop==True:
    Xtr = Xtrain[Xtrain.timestamp==rggg[j]]
    y = Xtr['y']
    Xtr=Xtr.drop('y',1)
    
    for i in np.random.randint(lo_lim,1000000000,2):
        tX_train0,tX_test0,ty_train0,ty_test0 = tts(Xtr,y,train_size=0.65,random_state = i)
        j=j+1
        #mdl0 = grid_search.GridSearchCV(mdl, param_grid=param_grid,cv=3, n_jobs=-1)
        #mdl0.fit(tX_train0, tX_train0).predict(tX_test0)
        prd=mdl.fit(tX_train0,ty_train0).predict(tX_test0)
        rrr2 =np.round(r2_score(ty_test0,prd),8)
        #print(rrr2)
        
        mdl_list.append(mdl)
        if rrr2 > rsqs:
            lo_lim=lo_lim+1000000 
            #print(tX_train0.shape)
            rsqs=rrr2
            model0=mdl
            mdl_list.append(mdl)
            r22 = (np.sign(rsqs)*np.sqrt(np.abs(rsqs)))
            print('R2 : ',rsqs,'r2_score : ',r22, 'i : ',i)
    tm = np.round(time.time() - start,4)
    lop = False if tm>=50 else True
    lop= False if j>=20 else lop
    j= 0 if j>= len(rggg)-2 else j 

print('time :',tm)

#print(len(cols))
print('pass')
#trains=trains.interpolate(method='akima')
#trains=trains.fillna(observation.features.median())
#trains = pd.DataFrame(data=prep.robust_scale(trains,axis=0,quantile_range=(0,100))
                            #,columns=(trains.columns))
def get_weighted_y(series):
    id, y = series["id"], series["y"]
    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y


k=0
prv_reward = 0;count=0;rew=0
while True:
    count=count+1
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    #timestamp=1587
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
    df_nona = observation.features.interpolate(method='akima')
    df_nona =df_nona[cols.drop('timestamp',1)]
    #indx_test =df_nona["timestamp"].unique()
    df_nona = df_nona.fillna(df_nona.median()) 
    #print(observation.features.shape)
    df_nona0 = pd.DataFrame(data=prep.robust_scale(df_nona,axis=0,quantile_range=(0,100))
                            ,columns=(df_nona.columns))
    df_nona0["timestamp"] = timestamp                   
    #print(df_nona0.head())
    #df_nona0 = pd.DataFrame(data=prep.StandardScaler(df_nona,axis=0,quantile_range=(0,100))
                            #,columns=(df_nona.columns))
    observation.target.y =mdl_list[k].predict( df_nona0)
    #model0.predict( df_nona0)#mdl_list[k].predict( df_nona0)#mdl.predict( df_nona0)# 
    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    k=k+1
    rew=rew+reward
    if k>=len(mdl_list)-1:
        k=0
    if reward>=prv_reward:
        #done=True
        print('you rare here',reward,done,timestamp)
    print(reward,done,timestamp)

    if done:
        print('my av_reward',rew/count)
        print("Public score: {}".format(info["public_score"]))
        break