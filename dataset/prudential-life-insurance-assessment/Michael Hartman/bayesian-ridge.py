"""
Created on Sat Feb  6 00:33:00 2016

@author: Michael Hartman

Inspired by: https://www.kaggle.com/mariopasquato/prudential-life-insurance-assessment/linear-model/code
"""

import pandas as pd
import numpy as np 
from scipy.optimize import fmin_powell
from sklearn.linear_model import BayesianRidge as BR
from sklearn.preprocessing import StandardScaler as SS
from ml_metrics import quadratic_weighted_kappa

"""
Scoring
"""

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)
    
    
def score_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score
    
def apply_offsets(data, offsets):
    for j in range(num_classes):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]
    return data
    
# global variables 
categorical =  ["Product_Info_1", "Product_Info_2", "Product_Info_3", 
                "Product_Info_5", "Product_Info_6", "Product_Info_7", 
                "Employment_Info_2", "Employment_Info_3", "Employment_Info_5",
                "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", 
                "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", 
                "InsuredInfo_7", "Insurance_History_1", "Insurance_History_2", 
                "Insurance_History_3", "Insurance_History_4", 
                "Insurance_History_7", "Insurance_History_8", 
                "Insurance_History_9", "Family_Hist_1", #"Medical_History_2", 
                "Medical_History_3", "Medical_History_4", "Medical_History_5", 
                "Medical_History_6", "Medical_History_7", "Medical_History_8", 
                "Medical_History_9",  "Medical_History_11", #"Medical_History_10",
                "Medical_History_12", "Medical_History_13", "Medical_History_14", 
                "Medical_History_16", "Medical_History_17", "Medical_History_18", 
                "Medical_History_19", "Medical_History_20", "Medical_History_21", 
                "Medical_History_22", "Medical_History_23", "Medical_History_25", 
                "Medical_History_26", "Medical_History_27", "Medical_History_28", 
                "Medical_History_29", "Medical_History_30", "Medical_History_31", 
                "Medical_History_33", "Medical_History_34", "Medical_History_35", 
                "Medical_History_36", "Medical_History_37", "Medical_History_38", 
                "Medical_History_39", "Medical_History_40", "Medical_History_41",
                'Product_Info_2_char', 'Product_Info_2_num']

label_column = 'Response'
columns_to_drop = ['Id', 'Response']
num_classes = 8

print("Load the data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# combine train and test
all_data = train.append(test)

# Columns below determined by previous linear regression
# See: https://www.kaggle.com/zeroblue/prudential-life-insurance-assessment/linear-model-0-65-in-lb
columns_to_drop_data = ["Product_Info_1",      "Product_Info_5",      "Product_Info_6",     
                        "Product_Info_7",      "Employment_Info_1",   "Employment_Info_2",  
                        "Employment_Info_4",   "Employment_Info_5",   "Employment_Info_6",  
                        "InsuredInfo_3",       "InsuredInfo_4",       "Insurance_History_5",
                        "Family_Hist_1",      "Medical_History_8",   "Medical_History_9",  
                        "Medical_History_10",  "Medical_History_16",  "Medical_History_21", 
                        "Medical_History_34",  "Medical_History_36",  "Medical_Keyword_1",  
                        "Medical_Keyword_4",   "Medical_Keyword_5",   "Medical_Keyword_7",  
                        "Medical_Keyword_8",   "Medical_Keyword_10",  "Medical_Keyword_12", 
                        "Medical_Keyword_13",  "Medical_Keyword_14",  "Medical_Keyword_16", 
                        "Medical_Keyword_17",  "Medical_Keyword_18",  "Medical_Keyword_20", 
                        "Medical_Keyword_21",  "Medical_Keyword_23",  "Medical_Keyword_24", 
                        "Medical_Keyword_27",  "Medical_Keyword_28",  "Medical_Keyword_29", 
                        "Medical_Keyword_32",  "Medical_Keyword_35",  "Medical_Keyword_36", 
                        "Medical_Keyword_42",  "Medical_Keyword_43",  "Medical_Keyword_44", 
                        "Medical_Keyword_46",  "Medical_Keyword_47",  "Medical_Keyword_48",                     
                        'Ht', 'Wt']

all_data.drop(columns_to_drop_data, axis=1, inplace=True)

print('Eliminate missing values')
# Make the test labels -1
all_data[label_column].fillna(-1, inplace=True)
# Fill in blanks
all_data = all_data.fillna(all_data.mean())

# Add custom variables. See https://www.kaggle.com/mariopasquato/prudential-life-insurance-assessment/linear-model/code
all_data['custom_var_1'] = all_data['Medical_History_15'] < 10
all_data['custom_var_3'] = all_data['Product_Info_4'] < 0.075
all_data['custom_var_4'] = all_data['Product_Info_4'] == 1
all_data['custom_var_6'] = (all_data['BMI'] + 1)**2
all_data['custom_var_7'] = all_data['BMI']**0.8
all_data['custom_var_8'] = all_data['Ins_Age']**8.5
all_data['BMI_Age'] = (all_data['BMI'] * all_data['Ins_Age'])**2.5
all_data['custom_var_10'] = all_data['BMI'] > np.percentile(all_data['BMI'], 0.8)
all_data['custom_var_11'] = (all_data['BMI'] * all_data['Product_Info_4'])**0.9
age_BMI_cutoff = np.percentile(all_data['BMI'] * all_data['Ins_Age'], 0.9)
all_data['custom_var_12'] = (all_data['BMI'] * all_data['Ins_Age']) > age_BMI_cutoff
all_data['custom_var_13'] = (all_data['BMI'] * all_data['Medical_Keyword_3'] + 0.5)**3
                    
# Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

# Get dummies for categorical data
categorical = np.setdiff1d(categorical, columns_to_drop_data)
all_data = pd.get_dummies(all_data, columns = categorical)
print(all_data.shape)

# Scale the data
data_columns = all_data.columns.difference(columns_to_drop)
scalar = SS()
all_data[data_columns] = scalar.fit_transform(all_data[data_columns])

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# split train and test
train = all_data[all_data[label_column]>0].copy()
test = all_data[all_data[label_column]<1].copy()

# Nonlinear transformation of the labels
hardcoded_values = np.array([-1.6, 0.7, 0.3, 3.15, 4.53, 6.5, 6.77, 9.0])
labels = hardcoded_values[train[label_column].values-1]

print("training model")
model = BR()
model.fit(train.drop(columns_to_drop, axis=1).values, labels)

print("Get initial predictions")
train_pred = model.predict(train.drop(columns_to_drop, axis=1).values)    
print('Train score is: ' + str(eval_wrapper(train_pred, train[label_column])))
test_pred = model.predict(test.drop(columns_to_drop, axis=1).values)    

print('Learn offsets')
offsets = np.array([0, -0.4, -0.5, -0.5, -0.25, -0.1, 0.3, 0])    
offset_preds = np.vstack((train_pred, train_pred, train[label_column].values))
offset_preds = apply_offsets(offset_preds, offsets)
prev_score = 0 
opt_order = [3, 4, 2, 5, 1, 6]
for j in opt_order:
    train_offset = lambda x: -score_offset(offset_preds, x, j) * 100
    offsets[j] = fmin_powell(train_offset, offsets[j], disp=False)

print('offsets:', offsets)  

offset_test_preds = np.vstack((test_pred, test_pred))
data = apply_offsets(offset_test_preds, offsets)

final_test_pred = np.round(np.clip(data[1], 1, 8)).astype(int)

preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_pred})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('linear_submission.csv')   





    

