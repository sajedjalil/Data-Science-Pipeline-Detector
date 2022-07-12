# Most of this script is taken from here : 
# https://www.kaggle.com/zeroblue/prudential-life-insurance-assessment/xgboost-with-optimized-offsets/code
# I just added imputation and scaling


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import Imputer, RobustScaler
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
from sklearn import cross_validation

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)
    
def get_params():
    
    params = {}
    params["objective"] = "reg:linear"     
    params["eta"] = 0.05
    params["min_child_weight"] = 5
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.67
    params["silent"] = 1
    params["max_depth"] = 6
    plst = list(params.items())

    return plst
    
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

print("Load the data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print("Size of the train data")
print(train.shape)

print("Get variables with a lot of (over 80%) missing values")
na_frames = train.isnull().sum()
na_thresh = train.shape[0]*0.8
print(na_frames[na_frames > na_thresh])

print("Drop columns with a lot of missing values")
train = train.drop(['Medical_History_10', 'Medical_History_24'], axis = 1)
test = test.drop(['Medical_History_10', 'Medical_History_24'], axis = 1)
print("Get other variables with missing values")
na_frames = train.isnull().sum()
na_frames[na_frames > 0]
# Transform missing variables by performing median imputation. 
# This is by no means a complete solution. But in this case I wanted to try something different.
# combine train and test
all_data = train.append(test)

# Found at https://www.kaggle.com/zeroblue/prudential-life-insurance-assessment/xgboost-with-optimized-offsets/code
# create any new variables    
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]
all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']
all_data['CF5'] = (all_data['BMI']*all_data['Product_Info_2'])
all_data['CF1'] = all_data['BMI']*all_data['Family_Hist_1']
all_data['CF2'] = all_data['BMI']*all_data['Medical_History_2']

# Use -1 for any others
all_data.fillna(-1, inplace=True)
med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

# split train and test
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()

response, trainID = train['Response'], train['Id']
train = train.drop(['Id','Response'], axis = 1)

testID = test['Id']
response_test = test['Response']
test = test.drop(['Id','Response'], axis = 1)

X_train, X_val, y_train, y_val = cross_validation.train_test_split(train, response, test_size=0.3, random_state=0)

train_new = X_train
test_new = test

print('Start modeling...')
num_classes = 8
xgb_num_rounds = 500
eta_list = [0.05] * 200 
eta_list = eta_list + [0.02] * 1000

xgtrain = xgb.DMatrix(train_new, y_train)
xgvalid = xgb.DMatrix(X_val, y_val)
xgtest = xgb.DMatrix(test_new)
watchlist  = [(xgvalid,'eval'), (xgtrain,'train')]

# get the parameters for xgboost
plst = get_params()
print(plst)      

model = xgb.train(plst, xgtrain, xgb_num_rounds, watchlist, learning_rates=eta_list) 

# get preds
valid_preds = model.predict(xgvalid, ntree_limit=model.best_iteration)
print('Validation score is:', eval_wrapper(valid_preds, y_val)) 
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
valid_preds = np.clip(valid_preds, -0.99, 8.99)
test_preds = np.clip(test_preds, -0.99, 8.99)

offsets = np.array([0.1, -0.5, -0.5, -0.5, -0.8, 0.02, 0.8, 1])
data = np.vstack((valid_preds, valid_preds, y_val))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
for j in range(num_classes):
    train_offset = lambda x: -apply_offset(data, x, j)
    offsets[j] = fmin_powell(train_offset, offsets[j])  

# apply offsets to test
data = np.vstack((test_preds, test_preds, response_test))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

preds_out = pd.DataFrame({"Id": testID.astype(int), "Response": final_test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('xgb_offset_submission.csv')
                