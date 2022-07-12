
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[3]:


import xgboost as xgb


# In[4]:

#use 10000 records for model training
pd.set_option("display.max_columns", 500)
train = pd.read_csv("../input/train_V2.csv")
train = train.sample(n=10000, random_state=1)


# In[5]:


train.head(10)


# In[6]:


train.shape




# In[8]:

#delete irrelevant columns which won't effect the winning percentage
del(train["Id"])
del(train["groupId"])
del(train["matchId"])
train.shape


# In[9]:


train.head(10)


# In[10]:


train['matchType'].value_counts()


# In[11]:

#ecode categorical variable "matchtype" into numeric value
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train['matchType'])
le.classes_


# In[12]:


array =le.transform(train['matchType'])
print(array)


# In[13]:


MatchType = pd.DataFrame(array)
print(MatchType)


# In[14]:

#delete the old categorical columns and replaced it with encoded column
del(train['matchType'])
train.insert(loc=12, column='matchType', value=MatchType)
print(train.head(10))


# In[15]:

#check if there is missing labled value before start training
train.loc[:,'winPlacePerc'].isnull().sum()



#create arrays for features to be train and target prediction
X,y = train.iloc[:,:-1], train.iloc[:,-1]


# In[22]:


#create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)


# In[23]:


#### Model 1: Classification tree with XGBoost
xg_cl = xgb.XGBClassifier(objective = 'reg:linear', n_estimator=10,seed=123)


# In[24]:


xg_cl.fit(X_train, y_train)


# In[25]:


preds = xg_cl.predict(X_test)


# In[26]:

#test model quality with accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" %(accuracy))


# In[27]:


#boosting with CV to select the best model
PUBG_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective": "reg:linear", "max_depth" :3}


# In[28]:


cv_results = xgb.cv(dtrain= PUBG_dmatrix, params=params, nfold=3, num_boost_round =5, metrics="error",as_pandas=True, seed=123)


# In[29]:


print(cv_results)


# In[30]:


print(((1-cv_results["test-error-mean"]).iloc[-1]))


# In[31]:


#test model quality with AUC under CV
from sklearn import metrics


# In[32]:


cv_results = xgb.cv(dtrain=PUBG_dmatrix, params=params, nfold=3, num_boost_round=5,metrics="auc",as_pandas=True,seed=123)
print(cv_results)


# In[33]:


print((cv_results["test-auc-mean"]).iloc[-1])


# In[34]:


#### Model 2: Regression tree with XGBoost
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)


# In[35]:


xg_reg= xgb.XGBRegressor(objective = "reg:linear", n_estimators=10, seed=123)


# In[36]:


xg_reg.fit(X_train, y_train)


# In[37]:


preds = xg_reg.predict(X_test)


# In[38]:

#test model with MSE 
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))


# In[39]:

#### Model 3: Linear Regression
#Linear Base Learner
DM_train = xgb.DMatrix(data= X_train, label= y_train)
DM_test = xgb.DMatrix(data= X_test, label=y_test)

params = {"booster":"gblinear", "objective":"reg:linear"}


# In[40]:


xg_linear = xgb.train(params=params, dtrain=DM_train, num_boost_round=10)


# In[41]:


preds = xg_linear.predict(DM_test)
preds.shape


# In[42]:

#test model with MSE
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# In[43]:


####Tuning Model with Linear Regression

pubg_dmatrix = xgb.DMatrix(data=X, label=y)

params = {"objective":"reg:linear", "max_depth":4}

cv_results = xgb.cv(dtrain=pubg_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="mae", as_pandas=True, seed=123)

print(cv_results)


# In[44]:


print((cv_results["test-mae-mean"]).tail(1))


# In[45]:


## Regularization in XGBoosting

pubg_dmatrix = xgb.DMatrix(data=X, label=y)
reg_params = [1,10,100]

params = {"objective":"reg:linear", "max_depth":3}

#create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

#interate over reg_params
for reg in (reg_params):
    params["lambda"] = reg
    cv_results_rmse = xgb.cv(dtrain=pubg_dmatrix, params=params, nfold=4, num_boost_round=5, metrics= "rmse",as_pandas=True, seed=123)
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])    

print(cv_results_rmse)


# In[46]:


print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2","rmse"]))



# In[47]:


## Tuning with boosting rounds
pubg_dmatrix = xgb.DMatrix(data = X, label = y)
params = {"objective":"reg:linear", "max_dpeth":3}
num_rounds = [5,10,15]
final_rmse_per_round = []
for curr_num_rounds in num_rounds:
    cv_results = xgb.cv(dtrain=pubg_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse",as_pandas=True, seed=123)
    
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])
print(cv_results)


# In[48]:


print(pd.DataFrame(list(zip(num_rounds, final_rmse_per_round)), columns=["num-boosting-rounds","rmse"]))


# In[51]:


## Auto boosting round selection using early_stopping
pubg_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:linear", "max_depth":4}
cv_results = xgb.cv(params = params, dtrain=pubg_dmatrix,metrics ="rmse", seed=123,num_boost_round = 50, early_stopping_rounds=10) 
print(cv_results)


# In[52]:


## Tuning Learning Rate
pubg_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:linear", "max_depth":3}
eta_vals = [0.001,0.01,0.1]
best_rmse = []

for curr_val in eta_vals:
    params["eta"] = curr_val
    
    cv_results = xgb.cv(dtrain=pubg_dmatrix, params=params, nfold=3, num_boost_round=10, early_stopping_rounds=5, metrics="rmse", as_pandas=True, seed=123)
    
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])


# In[53]:


print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns = ["eta","best_rmse"]))


# In[54]:


#### Grid Search with XGBoosting
from sklearn.model_selection import GridSearchCV
pubg_dmatrix = xgb.DMatrix(data=X, label=y)
gbm_param_grid = {
    'colsample_bytree': [0.3,0.7,0.9],
    'n_estomators':[50,100,150,200],
    'max_depth': [2,5,7],
    
    
}

gbm = xgb.XGBRegressor()

grid_mse = GridSearchCV(param_grid = gbm_param_grid, estimator=gbm, scoring = "neg_mean_squared_error", cv=4, verbose = 1)


# In[55]:


grid_mse.fit(X,y)


# In[56]:


print("Best parameters found:", grid_mse.best_params_)
print("Lowest RMSE found:", np.sqrt(np.abs(grid_mse.best_score_)))


# In[57]:

####RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
gbm_param_grid = {'n_estimators': [25],
                   'max_depth': range(2,12)
                 }
gbm = xgb.XGBRegressor(n_estimators=10)

randomized_mse = RandomizedSearchCV(param_distributions = gbm_param_grid,estimator = gbm, scoring = "neg_mean_squared_error", n_iter = 5, cv = 4, verbose = 1 )


# In[58]:


randomized_mse.fit(X,y)


# In[59]:


print("Best parameters found:", randomized_mse.best_params_)
print("Lowest RMSE found:", np.sqrt(np.abs(randomized_mse.best_score_)))


# In[88]:

####used the model built from GridSearch on test set for prediction
test = pd.read_csv("../input/test_V2.csv")


# In[89]:


test.shape


# In[90]:


test.head(20)


# In[94]:

#extract Id columns for future use in submission file
test_id = test.loc[:,"Id"]
print(test_id)


# In[75]:

#delete irrelevant categorical columns
del(test["Id"])
del(test["groupId"])
del(test["matchId"])
test.head(10)


# In[67]:

#encoded "matchType" columns with numeric variables
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test['matchType'])
le.classes_


# In[68]:


array_test =le.transform(test['matchType'])
print(array_test)


# In[69]:


MatchType_1 = pd.DataFrame(array_test)
print(MatchType_1)


# In[76]:

#replaced the categorical clumn with encoded numeric values
del(test['matchType'])
test.insert(loc=12, column='matchType', value=MatchType)
print(train.head(10))


# In[85]:

#applied grid_mse model built for prediction
pred_test = grid_mse.predict(test)


# In[86]:


print(pred_test)


# In[79]:


pred_test.shape


# In[87]:


pred_test_df = pd.DataFrame(pred_test, columns = ['winPlacePerc'])
print(pred_test_df)


# In[95]:

#insert Id column back to the final winPlacePerc prediction
pred_test_df.insert(loc=0, column='Id', value=test_id)


# In[96]:


print(pred_test_df)


# In[98]:


pred_test_df.to_csv('pred_test_submittion.csv', index = False)

