import pandas as pd
import os

os.system("ls ../input")

import numpy as np
from sklearn.cross_validation import train_test_split

data = pd.read_csv("../input/train.csv")


sub_data = pd.read_csv("../input/test.csv")

train, test = train_test_split(data, test_size = 0.2)

len(train)

len(test)

col_dict =  {}

for col in train.columns:
    col_dict.update({col:len(train[col].unique())})
    
    
### Random Forest
pd.DataFrame(col_dict,index= [1])

from sklearn.ensemble import RandomForestClassifier

rf_model_eq = RandomForestClassifier(n_estimators=500, criterion='gini',max_depth = 8, bootstrap=True, oob_score=True) 

rf_model = rf_model_eq.fit(train[[col for col in train.columns if col not in ('id','target')]],np.ravel(train[['target']]))

feat_imp = pd.DataFrame(rf_model.feature_importances_)
feat_imp.index =  [col for col in train.columns if col not in ('id','target')]
feat_imp.columns = ['feat_import']

train_data = train[[col for col in train.columns if col not in ('id','target')]]
rf_model = rf_model_eq.fit(train_data,np.ravel(train[['target']]))

test_predict = pd.DataFrame(rf_model.predict(test[[col for col in test.columns if col not in ('id','target')]]))
test_predict.columns = ['prediction']

test_final = pd.concat([test.reset_index(drop=True), test_predict], axis=1)
test_final['map'] = 0
test_final.ix[test_final.target == test_final.prediction,'map'] = 1
accuracy = (test_final.map.sum()/len(test_final))*100
accuracy

sub_predict = pd.DataFrame(rf_model.predict(sub_data[[col for col in sub_data.columns if col not in ('id','target')]]))
sub_predict.columns = ['prediction']

sub_final_rf = pd.concat([sub_data.reset_index(drop=True), sub_predict], axis=1)


req_feat =list(feat_imp[feat_imp.feat_import>=0.01].index)

rf_model_eq = RandomForestClassifier(n_estimators=250, criterion='gini',max_depth = 8, bootstrap=True, oob_score=True)

train_data = train[[col for col in train.columns if col in req_feat]]
rf_model = rf_model_eq.fit(train_data,np.ravel(train[['target']]))

test_predict = pd.DataFrame(rf_model.predict(test[[col for col in test.columns if col in req_feat]]))
test_predict.columns = ['prediction']

test_final = pd.concat([test.reset_index(drop=True), test_predict], axis=1)
test_final['map'] = 0
test_final.ix[test_final.target == test_final.prediction,'map'] = 1
accuracy = (test_final.map.sum()/len(test_final))*100
accuracy

sub_predict = pd.DataFrame(rf_model.predict(sub_data[[col for col in sub_data.columns if col in req_feat]]))
sub_predict.columns = ['prediction']

sub_final_rf1 = pd.concat([sub_data.reset_index(drop=True), sub_predict], axis=1)


## XGBoost
import xgboost

train_data = train[[col for col in train.columns if col not in ('id','target')]]

model = xgboost.XGBClassifier()
model.fit(train_data, np.ravel(train[['target']]))

test_predict = pd.DataFrame(model.predict(test[[col for col in test.columns if col not in ('id','target')]]))
test_predict.columns = ['prediction']

test_final = pd.concat([test.reset_index(drop=True), test_predict], axis=1)
test_final['map'] = 0
test_final.ix[test_final.target == test_final.prediction,'map'] = 1
accuracy = (test_final.map.sum()/len(test_final))*100
accuracy

sub_predict = pd.DataFrame(model.predict(sub_data[[col for col in sub_data.columns if col not in ('id','target')]]))
sub_predict.columns = ['prediction']

sub_final_xgb = pd.concat([sub_data.reset_index(drop=True), sub_predict], axis=1)


feat_df = model.booster().get_fscore()

feat_imp = pd.DataFrame(feat_df,columns=[key for key in feat_df], index = ['feat_import']).transpose()

req_feat =list(feat_imp[feat_imp.feat_import>=60].index)

train_data = train[[col for col in train.columns if col in req_feat]]

model = xgboost.XGBClassifier()
model.fit(train_data, np.ravel(train[['target']]))

test_predict = pd.DataFrame(model.predict(test[[col for col in test.columns if col  in req_feat]]))
test_predict.columns = ['prediction']

test_final = pd.concat([test.reset_index(drop=True), test_predict], axis=1)
test_final['map'] = 0
test_final.ix[test_final.target == test_final.prediction,'map'] = 1
accuracy = (test_final.map.sum()/len(test_final))*100
accuracy

sub_predict = pd.DataFrame(model.predict(sub_data[[col for col in sub_data.columns if col in req_feat]]))
sub_predict.columns = ['prediction']

sub_final_xgb1 = pd.concat([sub_data.reset_index(drop=True), sub_predict], axis=1)



submission = sub_final_xgb


submission['map_val'] = 1

submission = submission.pivot(index='id', columns='prediction', values='map_val').fillna(0)

submission[submission.columns] = submission[submission.columns].astype(int)

submission['id'] = submission.index

col_names = [col for col in submission.columns if col!='id']
col_names = ['id']+col_names


