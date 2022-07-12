import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import os



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
    params["min_child_weight"] = 240
    params["subsample"] = 1
    params["colsample_bytree"] = 0.71
    params["silent"] = 1
    params["max_depth"] = 7
    plst = list(params.items())

    return plst
    
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

# global variables
columns_to_drop = ['Id', 'Response', 'Medical_History_10','Medical_History_24']
#columns_to_drop = ['Id', 'Response']
xgb_num_rounds = 750
num_classes = 8
eta_list = [0.05] * 200 
eta_list = eta_list + [0.02] * 500
eta_list = eta_list + [0.01] * 100

print("Load the data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# combine train and test
all_data = train.append(test)


# Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
# create any new variables
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

print('Eliminate missing values')    
# Use -1 for any others
all_data.fillna(-1, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# split train and test
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()

# convert data to xgb data structure
xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)    

# get the parameters for xgboost
plst = get_params()
print(plst)      

# train model
model = xgb.train(plst, xgtrain, xgb_num_rounds, learning_rates=eta_list) 

# get preds
train_preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
print('Train score is:', eval_wrapper(train_preds, train['Response'])) 
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)


train_preds = np.clip(train_preds, -0.99, 8.99)
test_preds = np.clip(test_preds, -0.99, 8.99)


# train offsets 
offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
#offsets = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

data = np.vstack((train_preds, train_preds, train['Response'].values))

for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

for j in range(num_classes):
    train_offset = lambda x: -apply_offset(data, x, j)
    offsets[j] = fmin_powell(train_offset, offsets[j])  


# apply offsets to test
data = np.vstack((test_preds, test_preds, test['Response'].values))

for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('xgb_offset_submission.csv')

os.system('say "your program has finished"')
