#libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import random
from sklearn.ensemble import RandomForestRegressor
from scipy import stats




#Load Data


train = pd.read_csv('../input/train.csv')


test = pd.read_csv('../input/test.csv')



#ETL
M = train.copy()


train_ID = M.pop('ID')
train_target = M.pop('target')


test_ID = test.pop('ID')


#Scale (use MaxAbs to preserve sparcity)
max_abs_scaler = MaxAbsScaler()
M_scale = pd.DataFrame(max_abs_scaler.fit_transform(M),columns=M.columns)
test_X = pd.DataFrame(max_abs_scaler.transform(test),columns=test.columns)


#create fit/validate subsets
fit_X,validate_X,fit_y,validate_y = train_test_split(M_scale,train_target,test_size = 1/5)
fit_X.reset_index(inplace=True,drop=True)
validate_X.reset_index(inplace=True,drop=True)
fit_y.reset_index(inplace=True,drop=True)
validate_y.reset_index(inplace=True,drop=True)




#Eigenvector Approach

#Square matrix
fit_Q= np.matmul(fit_X.T,fit_X)

#calculate eigenpairs
eig_vals, eig_vecs = np.linalg.eig(fit_Q)

#function to iterate regressing various eigenvector spaces
def f(p) :

    #eigenvectors
    V = pd.DataFrame(np.real(eig_vecs[:,:p]))
    
    #train frame dimension reduction 
    T = np.matmul(fit_X,V) 
    
    #test frame dimension reduction
    t = np.matmul(validate_X,V) 
    
    #model (XGB)
    R = XGBRegressor(max_depth=10, learning_rate=0.005, n_estimators=10000, silent=True, objective='reg:linear'
    , booster='gbtree', n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0
    , subsample=0.7, colsample_bytree=0.5, colsample_bylevel=1, reg_alpha=0, reg_lambda=0, scale_pos_weight=1
    , base_score=0.5, random_state=0, seed=None, missing=None)# **kwargs) 
    #fit
    R.fit(T,np.log1p(fit_y),eval_set=[(t,np.log1p(validate_y))],early_stopping_rounds=100,verbose=200)

    #predict
    P = R.predict(t)
    #P = np.expm1(R.predict(t))

    #calculate metric error (RMSLE)
    E = np.sqrt( np.mean( np.power(P - np.log1p(validate_y), 2)  ) )

    #return metric error
    return E

#Approach_1 = {x : f(x) for x in range(10,101,10) }
#40




#Singular Value Decomposition Approach

#calculate SVD
u, s, vh = np.linalg.svd(fit_X)


#diagonalize s
Sigma = np.zeros((fit_X.shape[0],fit_X.shape[1]))
Sigma[:fit_X.shape[0],:fit_X.shape[0]] = np.diag(s)

#function to iterate regressing various singular value spaces
def g(p) :

    #singular values
    S = Sigma[:,:p]
    
    #train frame dimension reduction 
    T = u.dot(S)
    
    #test frame dimension reduction
    #t = validate_X.dot(vh[:,:p]) 
    t= np.matmul(validate_X,vh[:,:p])     
    
    #model (XGB)
    R = XGBRegressor(max_depth=10, learning_rate=0.005, n_estimators=10000, silent=True, objective='reg:linear'
    , booster='gbtree', n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0
    , subsample=0.7, colsample_bytree=0.5, colsample_bylevel=1, reg_alpha=0, reg_lambda=0, scale_pos_weight=1
    , base_score=0.5, random_state=0, seed=None, missing=None)# **kwargs) 
    #fit
    R.fit(T,np.log1p(fit_y),eval_set=[(t,np.log1p(validate_y))],early_stopping_rounds=100,verbose=200)

    #predict
    P = R.predict(t)
    #P = np.expm1(R.predict(t))

    #calculate metric error (RMSLE)
    E = np.sqrt( np.mean( np.power(P - np.log1p(validate_y), 2)  ) )

    #return metric error
    return E

#Approach_2 = {x : g(x) for x in range(10,101,10) }
#60




#CUR decomposition approach


#squared frobenius norm
CUR_Frobenius = np.power( np.linalg.norm(fit_X, ord='fro', axis=None, keepdims=False),2 )

#frobenius row/column probability distributions
CUR_col_indexes = np.array(range(len(fit_X.columns)))
CUR_col_probs = ([sum(np.power(fit_X.iloc[:,x],2)) / CUR_Frobenius for x in range(len(fit_X.columns))])
CUR_col_dist = stats.rv_discrete(name='CR_col_dist', values=(CUR_col_indexes, CUR_col_probs))

CUR_row_indexes = np.array(range(len(fit_X)))
CUR_row_probs = ([sum(np.power(fit_X.iloc[x,:],2)) / CUR_Frobenius for x in range(len(fit_X))])
CUR_row_dist = stats.rv_discrete(name='CR_row_dist', values=(CUR_row_indexes, CUR_row_probs))


#function to iterate regressing various row/column selections
#def h(p) :

#row/column selection
CUR_r = 70
CUR_col_select = CUR_col_dist.rvs(size=CUR_r)
CUR_row_select = CUR_row_dist.rvs(size=CUR_r)


#check for duplicates sampled
#len(pd.Series(CUR_col_select).unique()), len(pd.Series(CUR_row_select).unique())


#extract selected row/columns
CUR_C = fit_X.iloc[:,CUR_col_select]
CUR_W = fit_X.iloc[CUR_row_select,CUR_col_select]
CUR_R = fit_X.iloc[CUR_row_select,:]
 

#singular value decomposition of W
CUR_X,CUR_s,CUR_Y = np.linalg.svd(CUR_W)
#CUR_W.shape,CUR_X.shape,CUR_s.shape,CUR_Y.shape


#diagonalize CUR_s, calculate Moore-Penrose pseudo-inverse
CUR_sigma = np.zeros((CUR_W.shape[0],CUR_W.shape[1]))
CUR_sigma[:CUR_W.shape[0],:CUR_W.shape[0]] = np.diag( np.divide(1,CUR_s, where = CUR_s != 0) )



#calculate CUR_U
CUR_U = np.matmul(CUR_sigma , CUR_sigma)
CUR_U = np.matmul(CUR_Y , CUR_U)
CUR_U = np.matmul( CUR_U , CUR_X.T )
#CUR_U = pd.DataFrame(CUR_U)print(CUR_U)
#

#scaling by sqrt(rp)
CUR_col_scale = pd.Series(CUR_col_probs)
CUR_col_scale = CUR_col_scale[CUR_col_select]
CUR_col_scale = CUR_col_scale * CUR_r
CUR_col_scale = np.sqrt(CUR_col_scale)
CUR_C_scaled = pd.DataFrame( CUR_C.iloc[:,x] / CUR_col_scale.iloc[x] for x in range(CUR_r) ).T


CUR_row_scale = pd.Series(CUR_row_probs)
CUR_row_scale = CUR_row_scale[CUR_row_select]
CUR_row_scale = CUR_row_scale * CUR_r
CUR_row_scale = np.sqrt(CUR_row_scale)
CUR_R_scaled = pd.DataFrame( CUR_R.iloc[x,:] / CUR_row_scale.iloc[x] for x in range(CUR_r) )


#CUR

#CUR_CUR = np.matmul( CUR_C_scaled , CUR_U )
#CUR_CUR = np.matmul( CUR_CUR , CUR_R_scaled )
#CUR_CUR = pd.DataFrame(CUR_CUR)


#approximation error
#CURR_error = np.subtract(fit_X , CUR_CUR)
#CURR_error = (np.power( np.linalg.norm(CURR_error, ord='fro', axis=None, keepdims=False),2 ) )
#return CURR_error


#train frame dimension reduction 
T = CUR_C_scaled.dot(CUR_U)

#test frame dimension reduction
t= np.matmul(validate_X,CUR_R_scaled.T)     
t= pd.DataFrame(t)

#model (XGB)
R = XGBRegressor(max_depth=10, learning_rate=0.005, n_estimators=10000, silent=True, objective='reg:linear'
, booster='gbtree', n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0
, subsample=0.7, colsample_bytree=0.5, colsample_bylevel=1, reg_alpha=0, reg_lambda=0, scale_pos_weight=1
, base_score=0.5, random_state=0, seed=None, missing=None)# **kwargs) 
#fit
R.fit(T,np.log1p(fit_y),eval_set=[(t,np.log1p(validate_y))],early_stopping_rounds=100,verbose=200)

#predict
P = R.predict(t)
#P = np.expm1(R.predict(t))

#calculate metric error (RMSLE)
E = np.sqrt( np.mean( np.power(P - np.log1p(validate_y), 2)  ) )

#return metric error
#return E

#Approach_3 = {x : h(x) for x in range(10,101,10) }
#70




#Random Forest Approach


#model
RFR = RandomForestRegressor(n_jobs=-1)
#fit
RFR.fit(fit_X,np.log1p(fit_y))
#n_estimators=10, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1
#, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None
#, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False
#, n_jobs=1, random_state=None, verbose=0, warm_start=False)


#predict / calculate metric error (RMSLE)
np.sqrt( np.mean( np.power(RFR.predict(validate_X) - np.log1p(validate_y), 2)  ) )


#rank feature importance
RFR_importance = pd.DataFrame({'importance': RFR.feature_importances_, 'feature': [c for c in fit_X.columns]}).sort_values(by=['importance'], ascending=[False])


#top 10% features
len(RFR_importance)//10, sum(RFR_importance.iloc[:len(RFR_importance)//10,0])
#499
#0.81....




#*** FINAL *** selected model


fit_EV_features  = pd.DataFrame( np.matmul(fit_X, pd.DataFrame(np.real(eig_vecs[:,:40])) ),columns=[str(x)+'_EV' for x in range(40)] )

fit_SVD_features = pd.DataFrame( u.dot(Sigma[:,:60]),columns=[str(x)+'_SVD' for x in range(60)] )

fit_CUR_features = pd.DataFrame( T,columns=[str(x)+'_CUR' for x in range(70)] )

fit_RFR_features = fit_X[RFR_importance.iloc[:len(RFR_importance)//10,1]]

fit_X_final = pd.concat([fit_RFR_features,fit_EV_features,fit_SVD_features,fit_CUR_features],axis=1)


validate_EV_features  = pd.DataFrame( np.matmul(validate_X, pd.DataFrame(np.real(eig_vecs[:,:40])) ),columns=[str(x)+'_EV' for x in range(40)] )

validate_SVD_features = pd.DataFrame( np.matmul(validate_X,vh[:,:60]),columns=[str(x)+'_SVD' for x in range(60)] )

validate_CUR_features = pd.DataFrame( t,columns=[str(x)+'_CUR' for x in range(70)] )

validate_RFR_features = validate_X[RFR_importance.iloc[:len(RFR_importance)//10,1]]

validate_X_final = pd.concat([validate_RFR_features,validate_EV_features,validate_SVD_features,validate_CUR_features],axis=1)


model = XGBRegressor(max_depth=10, learning_rate=0.005, n_estimators=10000, silent=True, objective='reg:linear'
, booster='gbtree', n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0
, subsample=0.7, colsample_bytree=0.5, colsample_bylevel=1, reg_alpha=0, reg_lambda=0, scale_pos_weight=1
, base_score=0.5, random_state=0, seed=None, missing=None)# **kwargs) 
#fit
model.fit(fit_X_final,np.log1p(fit_y),eval_set=[(validate_X_final,np.log1p(validate_y))],early_stopping_rounds=100,verbose=200)
#[0]     validation_0-rmse:13.9775
#Will train until validation_0-rmse hasn't improved in 100 rounds.
#[200]   validation_0-rmse:5.37171
#[400]   validation_0-rmse:2.45208
#[600]   validation_0-rmse:1.64308
#[800]   validation_0-rmse:1.47544
#[1000]  validation_0-rmse:1.44293
#[1200]  validation_0-rmse:1.4363
#[1400]  validation_0-rmse:1.4337
#[1600]  validation_0-rmse:1.43271
#[1800]  validation_0-rmse:1.43193
#[2000]  validation_0-rmse:1.43132
#[2200]  validation_0-rmse:1.43111
#[2400]  validation_0-rmse:1.4307
#[2600]  validation_0-rmse:1.43047
#[2800]  validation_0-rmse:1.43022
#Stopping. Best iteration:
#[2777]  validation_0-rmse:1.43019


#predict
validate_prediction = model.predict(validate_X_final)


#calculate metric error (RMSLE)
print( np.sqrt( np.mean( np.power(validate_prediction - np.log1p(validate_y), 2) ) ) )
# 1.4302381613970911




#solving


#features
test_EV_features  = pd.DataFrame( np.matmul(test_X, pd.DataFrame(np.real(eig_vecs[:,:40])) ),columns=[str(x)+'_EV' for x in range(40)] )

test_SVD_features = pd.DataFrame( np.matmul(test_X,vh[:,:60]),columns=[str(x)+'_SVD' for x in range(60)] )

test_CUR_features = pd.DataFrame( np.matmul(test_X,CUR_R_scaled.T) ,columns=[str(x)+'_CUR' for x in range(70)] )

test_RFR_features = test_X[RFR_importance.iloc[:len(RFR_importance)//10,1]]

test_X_final = pd.concat([test_RFR_features,test_EV_features,test_SVD_features,test_CUR_features],axis=1)


#predict
test_prediction = model.predict(test_X_final)


#output
y_hat = pd.DataFrame({'ID': test_ID, 'target': np.expm1(test_prediction)})
y_hat.to_csv('kaggle_santander_6_29_18.csv', index=False)
#LB: 1.53

