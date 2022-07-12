import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

test['Type']='Test'
train['Type']='Train'

MergedData = pd.concat([test,train],axis=0)
MergedData.columns

ID_col = ['Id']
target_col = ['Response']
categorical_cols = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']
num_cols = list(set(list(MergedData.columns))-set(categorical_cols)-set(ID_col)-set(target_col))
other_cols = ['Type']

MergedData.isnull().any()

NumCategorical = num_cols+categorical_cols

for var in NumCategorical:
    if MergedData[var].isnull().any()==True:
        MergedData[var+'_NA']=MergedData[var].isnull()*1
        
MergedData[num_cols] = MergedData[num_cols].fillna(MergedData[num_cols].mean(),inplace=True)

for var in categorical_cols:
 number = LabelEncoder()
 MergedData[var] = number.fit_transform(MergedData[var].astype('str'))

MergedData["Response"] = number.fit_transform(MergedData["Response"].astype('str'))

train=MergedData[MergedData['Type']=='Train']
test=MergedData[MergedData['Type']=='Test']

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

features=list(set(list(MergedData.columns))-set(ID_col)-set(target_col)-set(other_cols))

train1 = Train[list(features)].values
train2 = Train["Response"].values
validate1 = Validate[list(features)].values
validate2 = Validate["Response"].values
test1=test[list(features)].values      

random.seed(100)
RandomForestCl = RandomForestClassifier(n_estimators=1000)
RandomForestCl.fit(train1, train2) 

final = RandomForestCl.predict_proba(test1)
test["Response"]=final[:,1]
test.to_csv('model_output.csv',columns=['Id','Response'])