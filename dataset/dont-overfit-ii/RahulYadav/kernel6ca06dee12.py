import matplotlib.pyplot as plt
import seaborn as sns

# lib to read data and mathematical operations 
import pandas as pd
import numpy as np

# Libaries for featureengg. and ML
# Preprocessing Scaling features
from sklearn.preprocessing import RobustScaler
# library for feature selection
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
# for ml model to be used for feature selection
from sklearn.linear_model import LogisticRegression,Lasso
# metrics for evaluate the prediction
from sklearn.metrics import roc_auc_score, accuracy_score, make_scorer

# for training model
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score




# FE

n_fold = 20
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=213)
threshold = 0.80



def scoring_roc_auc(y, y_pred):
        try:
            return roc_auc_score(y, y_pred)
        except:
            return 0.5

robust_roc_auc = make_scorer(scoring_roc_auc)
    
    
def train_model(X, X_tests, y, params, text,folds=folds,model=None,threshold=0.80,feature_selection=True):
    prediction = np.zeros(len(X_tests))
    scores = []
#     accuracy=[]
    if feature_selection:
        grid_search = GridSearchCV(model, param_grid=params, verbose=0, n_jobs=-1, scoring=robust_roc_auc, cv=20)    
        grid_search.fit(X,y)
        feature_selector = RFE(grid_search.best_estimator_, n_features_to_select=12,step=15, verbose=0)

#     if verbose==1    
#     print("~"*120)
#     print('\t\t\t\t{}_applied_ai_data'.format(text))
#     print('-'*120,'\n')
#     print('\t\t\tVal. scores for each folds and stacking status...')
#     print('-'*120)
#     print("\t\t\t\t\t fold |  val_roc   ")
#     print("\t\t\t\t\t-------------------")
    
        
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        # print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        if feature_selection:
            feature_selector.fit(X_train, y_train)
            X_train  = feature_selector.transform(X_train)
            X_valid  = feature_selector.transform(X_valid)
            X_test   = feature_selector.transform(X_tests)
            model    = feature_selector.estimator_
        
        
        grid_search = GridSearchCV(model, param_grid=params, n_jobs=-1, scoring=robust_roc_auc, cv=20)
        grid_search.fit(X_train, y_train)

            
        model = grid_search.best_estimator_
        
        model.fit(X_train, y_train)
        y_pred_valid = model.predict(X_valid).reshape(-1,)
   
        val_roc = roc_auc_score(y_valid, y_pred_valid)
#         val_acc  = accuracy_score(y_valid, y_pred_valid)
        

        if val_roc > threshold:
            message = '<-- OK - Stacking'
            try:
                y_pred = model.predict_proba(X_test)[:, 1]
            except:
                y_pred = model.predict(X_test)

            scores.append(roc_auc_score(y_valid, y_pred_valid))
#             accuracy.append(accuracy_score(y_valid, y_pred_valid))
            prediction += y_pred
            
        else:
            message = '<-- skipping'
            
        print("\t\t\t\t\t {:2}   |  {:.4f}       \t{}   ".format(fold_n, val_roc,message))
    
    
    
    prediction /= n_fold
    if prediction.sum()>0:
        print('-'*50)
        print('CV mean score of model after folds: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        print('')
#         sub = pd.read_csv('../input/dont-overfit-ii/sample_submission.csv')
#         sub['target']=prediction
#         sub.to_csv('{}_1.csv'.format(text),index=False)
        
#         print('\n Result : Created Submission file - "{}_applied_ai_data.csv"'.format(text))
        print('_'*120,'\n\n')
    else:
        print('\n Results Discarding the current ML agorithm - because Threshod cretria not meet')
        print('_'*120,'\n\n')
    
    return prediction, scores




def train_stack_models(X, X_tests, y,text, params1=None,params2=None, model1=None,model2=None):
    prediction = np.zeros(len(X_tests))

    pred1,sc1 = train_model(X,X_tests,y,params1,'Model1', model=model1,feature_selection=True)
    pred2,sc2 = train_model(X,X_tests,y,params2,'Model2', model=model2,feature_selection=True)
      
    
    prediction = (pred1+pred2)/2
    if prediction.sum()>0:
        print('-'*50)
        print('')
        sub = pd.read_csv('../input/dont-overfit-ii/sample_submission.csv')
        sub['target']=prediction
        sub.to_csv('Submission.csv'.format(text),index=False)
        
        print('\n Result : Created Submission file - "{}_applied_ai_data.csv"'.format(text))
        print('_'*120,'\n\n')
    else:
        print('\n Results Discarding the current ML agorithm - because Threshod cretria not meet')
        print('_'*120,'\n\n')
    
    return prediction, sc1,sc2




# reading the training data
# preparing the data
data = pd.read_csv('../input/dont-overfit/train.csv')
# Creating label matrics
target = data.target.values.astype(int)
# removing id and target coulumns for preparing training data 
train_data = data.drop(columns=['id','target'])

# Test Data
# reading and creating test data
test_data = pd.read_csv('../input/dont-overfit/test.csv')
test_data = test_data.drop(columns='id')

# taking correlated data
corrs = data.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
corrs = corrs[corrs['level_0'] == 'target']
# taking top 30 features coreelated with target
target_corr = list(corrs.level_1[-32:-1].values)
target_corr.remove('id')
# print(target_corr)
col =list(train_data.columns)
# resacaling data
rs_scaler = RobustScaler()
rs_scaler = rs_scaler.fit(train_data[col])
Xtrain = rs_scaler.transform(train_data[col])
Xtest = rs_scaler.transform(test_data[col])
Xtrain=pd.DataFrame(Xtrain,columns=col)
Xtest=pd.DataFrame(Xtest,columns=col)

param_model1 ={"C":[0.01,0.1,0.2, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37], "penalty":["l1"],
               'tol'   : [0.0001, 0.00011, 0.00009],'solver':['liblinear'],'max_iter':[500]}

param_model4 = {'alpha' : [0.022, 0.021, 0.02, 0.019, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
              'tol'   : [0.0013, 0.0014, 0.001, 0.0015, 0.0011, 0.0012, 0.0016, 0.0017]}




random_state=42
model1 = LogisticRegression(class_weight='balanced',random_state=random_state) 
model4 = Lasso(alpha=0.031, tol=0.01, random_state=random_state, selection='random')


# Xtrain.head()
_,s18a1,s18a2 = train_stack_models(Xtrain[target_corr].values,Xtest[target_corr].values,target, text='Submission',
                                   params1=param_model1,params2=param_model4,
                                   model1=model1,model2=model4)