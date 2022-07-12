import numpy as np 
import pandas as pd 
import lightgbm as gbm
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

print("="*50)
print("Load Data")
app_train=pd.read_csv("../input/application_train.csv")
app_test=pd.read_csv("../input/application_test.csv")
prev = pd.read_csv('../input/previous_application.csv')

print("Application Data")
print(app_train.shape)
print(app_test.shape)
print(prev.shape)


Target=app_train.TARGET.copy()
app_train.drop(labels=['TARGET'],inplace=True,axis=1)

print("="*50)
print("Processing Categories Features")
Var_Cat=app_train.select_dtypes('object').columns.tolist()

lb=LabelEncoder()
Cat_dict={}

for col in Var_Cat:
    app_train[col]=app_train[col].astype('str')
    app_test[col]=app_test[col].astype('str')

for col in Var_Cat:
    Cat_dict[col]=np.concatenate((app_train[col].unique(),app_test[col].unique()),axis=0)

for col in Var_Cat:
    lb.fit(Cat_dict[col])
    app_train[col]=lb.transform(app_train[col])
    app_test[col]=lb.transform(app_test[col])

del Cat_dict    

print(app_train.shape)
print(app_test.shape)

prev_cat_features = [f_ for f_ in prev.columns if prev[f_].dtype == 'object']
for f_ in prev_cat_features:
    prev[f_], _ = pd.factorize(prev[f_])
    
avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']

app_train = app_train.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
app_test = app_test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')


print("="*50)
print("Train Simple Model")
print(app_train.shape)
print(app_test.shape)


app_train.drop(labels=['SK_ID_CURR'],axis=1,inplace=True)
app_test.drop(labels=['SK_ID_CURR'],axis=1,inplace=True)

#X_train,X_val,y_train,y_val=train_test_split(app_train, Target, test_size=0.3, random_state=23)

#dtrain =gbm.Dataset(data = X_train, label = y_train)
#dval =gbm.Dataset(data = X_val, label = y_val)

#del(X_train,X_val, y_train,y_val)

Dparam = {'objective' : 'binary',
          'boosting_type': 'gbdt',
          'metric' : 'auc',
          'nthread' : 4,
          'shrinkage_rate':0.01,
          'max_depth':18,
          'min_child_weight': 2,
          'bagging_fraction':0.65,
          'feature_fraction':0.8,
          'lambda_l1':1,
          'lambda_l2':1,
          'num_leaves':35}        

folds = KFold(n_splits=5, shuffle=True, random_state=546789)
import gc
oof_preds = np.zeros(app_train.shape[0])
sub_preds = np.zeros(app_test.shape[0])
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(app_train)):
    dtrain =gbm.Dataset(app_train.iloc[trn_idx], Target.iloc[trn_idx])
    dval =gbm.Dataset(app_train.iloc[val_idx], Target.iloc[val_idx])
    m_gbm=gbm.train(params=Dparam,train_set=dtrain,num_boost_round=2000,verbose_eval=1000,valid_sets=[dtrain,dval],valid_names=['train','valid'])
    oof_preds[val_idx] = m_gbm.predict(app_train.iloc[val_idx])
    sub_preds += m_gbm.predict(app_test) / folds.n_splits
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(Target.iloc[val_idx],oof_preds[val_idx])))
    del dtrain,dval
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(Target, oof_preds))   


#Pred=m_gbm.predict(app_test)
print("Output Model")
Submission=pd.read_csv("../input/sample_submission.csv")
Submission['TARGET']=sub_preds.copy()
Submission.to_csv("baseline_gbm_Base.csv", index=False)    
    

