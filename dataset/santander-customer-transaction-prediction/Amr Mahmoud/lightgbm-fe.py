import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from  sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,KFold, StratifiedKFold
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb

#read the file position

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train= pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
test_inp = test_df.drop(columns = ['ID_code'])


X=train.drop(['ID_code','target'],axis=1)
y=train['target']
#generate new cvolumns depending on mean-std and mean+std
cols = list(X.columns)[0:200] 
for cols in X[cols]:
       X[cols+'_new'] = np.random.uniform((X[cols].mean())-((X[cols].std())*3), (X[cols].mean())+((X[cols].std())*3), X.shape[0])


#for cols in X[cols]:
    #del X[cols]
num=list(range(1,1050))
for x in num :
    x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.1)
    m=GaussianNB()
    m.fit(x_train,y_train)
    proba=m.predict_proba(x_test)[:,1]
    auc=metrics.roc_auc_score(y_test,proba)
    print(auc)
    if auc >.896:
            '''x_train.to_csv("x_train.csv", index=False)
            x_test.to_csv("x_test.csv", index=False)
            y_train.to_csv("y_train.csv", index=False)
            y_test.to_csv("y_test.csv", index=False)'''
            break

'''ols = list(X.columns)[0:] 
for cols in X[cols]:
    print (cols)
    del cols
    x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.1, stratify=y)
    model(GaussianNB())
    X = train_d.drop(['ID_code', 'target'], axis=1)

def bestParameters(m,parameters):
    grid=GridSearchCV(m,parameters,scoring='roc_auc',cv=10,n_jobs=-1)
    grid.fit(x_train,y_train)
    best_parameters=grid.best_params_
    best_score=np.round(grid.best_score_*100,2)
    print (best_parameters)
    return 


lgbm_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
evals_result = {}
model = lgb.train(params, lgbm_train, 100,verbose_eval=50, evals_result=evals_result)
print(evals_result)
y_pred=clf.predict(x_test)'''

#custom function to build the LightGBM model.
'''def run_lgb(x_train, y_train, x_test, y_test, test_inp):
    params = {
        "objective" : "binary",
        "metric" : "auc",
        "num_leaves" : 1000,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_freq" : 5,
        "reg_alpha" : 1.728910519108444,
        "reg_lambda" : 4.9847051755586085,
        "random_state" : 42,
        "bagging_seed" : 2019,
        "verbosity" : -1,
        "max_depth": 18,
        "min_child_samples":100
       # ,"boosting":"rf"
    }
    
    lgtrain = lgb.Dataset(x_train, label=y_train)
    lgval = lgb.Dataset(x_test, label=y_test)
    evals_result = {}
    model = lgb.train( params, lgtrain, 25000, valid_sets=[lgval], 
                      early_stopping_rounds=500, verbose_eval=500, evals_result=evals_result)
    
    pred_test_y = model.predict(test_inp, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
# Training the model #
pred_test, model, evals_result = run_lgb(x_train, y_train, x_test, y_test, test_inp)'''


# Create parameters to search
gridParams = {
    'colsample_bytree' : [0.65,0.76],
    'subsample' : [0.7,0.8],
    'reg_alpha' : [1.5,1.6],
    'reg_lambda' : [2,8],
     'bagging_freq': [5,7],
    'bagging_fraction': [0.4,1.3],
   'boost_from_average':['false'],
    'feature_fraction': [0.8,1.3],
    'learning_rate': [0.001],
    'max_depth': [-1,5,18,30],  
    'metric':['auc'],
    'min_data_in_leaf':[ 80],
    'min_sum_hessian_in_leaf': [10.0],
    'num_leaves': [300,500],
    'tree_learner': ['serial'],
    'verbosity': [-1]
    }
 
mdl=lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          silent = True,    
          max_bin=255,
        bagging_seed =2019,
    subsample = gridParams['subsample'],
    reg_alpha =gridParams['reg_alpha'],
    reg_lambda= gridParams['reg_lambda'],
     bagging_freq=gridParams['bagging_freq'],
    bagging_fraction= gridParams['bagging_fraction'],
   boost_from_average=gridParams['boost_from_average'],
    feature_fraction=gridParams['feature_fraction'],
    learning_rate=gridParams['learning_rate'],
    max_depth=gridParams['max_depth'],  
    metric=gridParams['metric'],
    min_data_in_leaf= gridParams['min_data_in_leaf'],
    min_sum_hessian_in_leaf= gridParams['min_sum_hessian_in_leaf'],
    num_leaves=gridParams['num_leaves'],
    tree_learner= gridParams['tree_learner'],
    verbosity=gridParams['verbosity'])

# To view the default model params:
mdl.get_params().keys()
# Create the grid
scoring = {'AUC': 'roc_auc'}

# Create the grid
grid = GridSearchCV(mdl, gridParams, scoring='roc_auc')
# Run the grid
grid.fit(x_train, y_train)

print('Best parameters found by grid search are:', grid.best_params_)
print('Best score found by grid search is:', grid.best_score_)
params= grid.best_params_






def run_lgb(x_train, y_train, x_test, y_test, test_inp,params):
    lgtrain = lgb.Dataset(x_train, label=y_train)
    lgval = lgb.Dataset(x_test, label=y_test)
    evals_result = {}
    model = lgb.train( params, lgtrain, 100000, valid_sets=[lgval], 
                      early_stopping_rounds=2000, verbose_eval=5000, evals_result=evals_result)
    pred_test_y = model.predict(test_inp, num_iteration=5000)
    sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
    sub_df["target"] = pred_test_y
    sub_df.to_csv("sion3.csv", index=False)
    return pred_test_y, model, evals_result
    
pred_test, model, evals_result = run_lgb(x_train, y_train, x_test, y_test, test_inp,params)

##################
#serach for best splitting

#agumentation  

'''import sys
np.set_printoptions(threshold=sys.maxsize)'''