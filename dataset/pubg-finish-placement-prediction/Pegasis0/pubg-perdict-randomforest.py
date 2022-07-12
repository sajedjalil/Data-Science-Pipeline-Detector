import numpy as np
import pandas as pd
import os
import gc
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz

def change_columns_to_num(input_data,columns_from,columns_new):
    for column in columns_new:
        input_data[column]=np.where(input_data[columns_from]==column,1,0)
    input_data=input_data.drop(columns_from,axis=1)
    return input_data

csv_path="../input/train_V2.csv"
data=pd.read_csv(csv_path)
data.describe()
data.isnull().sum()
data=data.dropna(axis=0) 
#记得注释掉↓
#data=data.sample(n=1000000)

y=data.winPlacePerc
X=data.copy()
X=change_columns_to_num(X,"matchType",["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"])
X=X.drop(["Id","groupId","matchId","winPlacePerc","longestKill"],axis=1)
del data
print (gc.collect())

'''
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1,test_size=0.2)

#model = RandomForestRegressor(n_estimators=20,n_jobs=-1,random_state=1,verbose=3,max_features=0.5, min_samples_leaf=3,oob_score=False)
#min_samples_leaf=5
#max_features="auto"
#max_leaf_nodes=-1
#min_samples_split=2
#max_depth=20
#oob_score=False


model = RandomForestRegressor(n_estimators=40,n_jobs=-1,verbose=3, min_samples_leaf=4,oob_score=False,
max_features="auto",max_depth=20,min_samples_split=2,max_leaf_nodes=-1)
model.fit(train_X,train_y)

predict_result=model.predict(val_X)
MAE=mean_absolute_error(val_y,predict_result)
del model
gc.collect()
print ("MAE="+str(MAE))


'''
final_model = RandomForestRegressor(n_estimators=100,n_jobs=-1,verbose=3, min_samples_leaf=4,oob_score=False,
max_features="auto",max_depth=20,min_samples_split=2,max_leaf_nodes=-1)
final_model.fit(X,y)
del X
del y
print (gc.collect())

test_data_path = '../input/test_V2.csv'
test_data=pd.read_csv(test_data_path)
test_data_Id=test_data.Id

test_X=change_columns_to_num(test_data,"matchType",
                      ["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"])
test_X=test_X.drop(["Id","groupId","matchId","longestKill"],axis=1)
test_X.assign(ka=(test_X.kills+test_X.assists/2))
del test_data
print (gc.collect())

test_predict_result=final_model.predict(test_X)

output = pd.DataFrame({'Id': test_data_Id,
                       'winPlacePerc': test_predict_result})
output.to_csv('submission.csv', index=False)