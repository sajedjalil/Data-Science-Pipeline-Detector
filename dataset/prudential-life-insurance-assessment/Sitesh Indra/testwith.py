import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_slsqp
from ml_metrics import quadratic_weighted_kappa


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

# global variables
columns_to_drop = ['Id', 'Response', 'Medical_History_1']
xgb_num_rounds = 10
num_classes = 8
eta_list = [0.5]*10
#eta_list = [0.05] * 100 
#eta_list = eta_list + [0.02] * 700

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
# Thanks to inversion for the idea to use np.digitize
# https://www.kaggle.com/inversion/prudential-life-insurance-assessment/digitize/discussion
train_offset = lambda x: -eval_wrapper(np.digitize(train_preds, x), train['Response'])
splits = np.array([0, 2, 2.8, 3.2, 4, 4.67, 5.75, 6.5])
cons = lambda x: np.array([x[i+1] - x[i] for i in range(len(x)-1)])
bnds = [(0, 8)] * 8
splits = fmin_slsqp(train_offset, splits, bounds=bnds, f_ieqcons=cons)
print('splits:', splits)
offset_train_score = eval_wrapper(np.digitize(train_preds, splits), train['Response'])
print('Offset train score is:', offset_train_score) 

# apply offsets to test
test_preds = np.digitize(test_preds, splits)

final_test_preds = np.round(np.clip(test_preds, 1, 8)).astype(int)

preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('xgb_offset_submission.csv')