# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy import stats
from numpy import loadtxt
from scipy.optimize import fmin_powell
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# get useful function to EDA
def feature_summary(df_fa):
    print('DataFrame shape')
    print('rows:',df_fa.shape[0])
    print('cols:',df_fa.shape[1])
    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']
    df=pd.DataFrame(index=df_fa.columns,columns=col_list)
    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])
    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])
    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])
    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])
    for i,col in enumerate(df_fa.columns):
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))
            df.at[col,'Mean']=df_fa[col].mean()
            df.at[col,'Std']=df_fa[col].std()
            df.at[col,'Skewness']=df_fa[col].skew()
        df.at[col,'Sample_values']=list(df_fa[col].unique())
           
    return(df.fillna('-'))
    
# loading the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
    
# define evaluation matric
def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return cohen_kappa_score(yhat, y, weights= 'quadratic')

#def get_params():
    
#    params = {}
#    params["objective"] = "reg:linear"     
#    params["eta"] = 0.01
#    params["min_child_weight"] = 300
#    params["subsample"] = 0.95
#    params["colsample_bytree"] = 0.4
#    params["silent"] = 1
#    params["max_depth"] = 15
#    plst = list(params.items())

#    return plst

def get_params():
    
    params = {}
    params["objective"] = "reg:linear"     
    params["eta"] = 0.02
    params["min_child_weight"] = 360
    params["subsample"] = 0.95
    params["colsample_bytree"] = 0.3
    params["silent"] = 1
    params["max_depth"] = 10
    plst = list(params.items())

    return plst

    
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
columns_to_drop = ['Id', 'Response'] #, 'Medical_History_10','Medical_History_24']
xgb_num_rounds = 800
num_classes = 8
missing_indicator = -1000

#get the data
train = df_train
test = df_test
# combine train and test
all_data = train.append(test)

from IPython.display import display
pd.options.display.max_columns = None

display(feature_summary(all_data))

# EDA skipped in this notebook

# Feature
# combine train and test
all_data = train.append(test)

# old version
# create any new variables    
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]
# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

# encoding with Product_Info_2
#le = LabelEncoder()
#le.fit(all_data['Product_Info_2'])
#all_data['Product_Info_2'] = le.transform(all_data['Product_Info_2'])

# frequency encoding for the most important features
#encoding = all_data.groupby('Medical_History_4').size()
#encoding = encoding/len(all_data)
#all_data['Medical_History_4'] = all_data['Medical_History_4'].map(encoding)

#encoding = all_data.groupby('Medical_History_15').size()
#encoding = encoding/len(all_data)
#all_data['Medical_History_15'] = all_data['Medical_History_15'].map(encoding)

#encoding = all_data.groupby('Medical_History_40').size()
#encoding = encoding/len(all_data)
#all_data['Medical_History_40'] = all_data['Medical_History_40'].map(encoding)

#encoding = all_data.groupby('Medical_History_30').size()
#encoding = encoding/len(all_data)
#all_data['Medical_History_30'] = all_data['Medical_History_30'].map(encoding)

#encoding = all_data.groupby('Medical_History_23').size()
#encoding = encoding/len(all_data)
#all_data['Medical_History_23'] = all_data['Medical_History_23'].map(encoding)


all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

print('Eliminate missing values')    
all_data.fillna(missing_indicator, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# after feature generation
display(feature_summary(all_data))

# Modeling
# split train and test
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()

# convert data to xgb data structure
xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values, 
                        missing=missing_indicator)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values, 
                        missing=missing_indicator)    

# get the parameters for xgboost
plst = get_params()
print(plst)      

# train model
model = xgb.train(plst, xgtrain, xgb_num_rounds) 

# get preds
train_preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
print('Train score is:', eval_wrapper(train_preds, train['Response'])) 
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)

# train offsets 
offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
offset_preds = np.vstack((train_preds, train_preds, train['Response'].values))
offset_preds = apply_offsets(offset_preds, offsets)
opt_order = [6,4,5,3]
for j in opt_order:
    train_offset = lambda x: -score_offset(offset_preds, x, j) * 100
    offsets[j] = fmin_powell(train_offset, offsets[j], disp=False)

print('Offset Train score is:', eval_wrapper(offset_preds[1], train['Response'])) 

# apply offsets to test
data = np.vstack((test_preds, test_preds, test['Response'].values))
data = apply_offsets(data, offsets)

final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('xgb_offset_submission.csv')