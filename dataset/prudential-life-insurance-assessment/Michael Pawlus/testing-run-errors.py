print("read in libraries")
import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa

print("define eval")
def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)
    
print("define parameters")    
def get_params():
    
    params = {}
    params["objective"] = "reg:linear"     
    params["eta"] = 0.05
    params["min_child_weight"] = 240
    params["subsample"] = 0.9
    params["colsample_bytree"] = 0.67
    params["silent"] = 0
    params["max_depth"] = 6
    params["max_delta_step"] = 1
    plst = list(params.items())

    return plst

print("define offset")       
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

print("define global variables")  
# global variables
columns_to_drop = ['Id', 'Response', 'Medical_History_10','Medical_History_24']
xgb_num_rounds = 800
num_classes = 8
eta_list = [0.1] * 100 
eta_list = eta_list + [0.05] * 200 
eta_list = eta_list + [0.025] * 300
eat_list = eta_list + [0.01] * 200

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

# combine train and test
print("combine")
all_data = train.append(test)

# create any new variables
print("new vars")
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

# factorize categorical variables
print("factors")
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

## combined features
# BMI by Age
print("BMI by Age")
all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']
all_data['Ins_Age_sq'] = all_data['Ins_Age'] * all_data['Ins_Age']
all_data['Ht_sq'] = all_data['Ht'] * all_data['Ht']
all_data['Wt_sq'] = all_data['Wt'] * all_data['Wt']
all_data['Ins_Age_cu'] = all_data['Ins_Age'] * all_data['Ins_Age'] * all_data['Ins_Age']
all_data['Ht_cu'] = all_data['Ht'] * all_data['Ht'] * all_data['Ht']
all_data['Wt_cu'] = all_data['Wt'] * all_data['Wt'] * all_data['Wt']

## summed features
# med keyword sum
print("med keyword sum")
med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

print('Eliminate missing values')    
# Use -1 for NAs
all_data.fillna(-1, inplace=True)

# fix the dtype on the label column (convert to integer)
print('response conversion')   
all_data['Response'] = all_data['Response'].astype(int)

# split train and test
print('split') 
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()

# convert data to xgb data structure
print('matrix formation') 
xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), label=train['Response'].values)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)   

# get the parameters for xgboost
print('parameters') 
plst = get_params()
print(plst)

# train model
print('train model') 
model = xgb.train(plst, xgtrain, xgb_num_rounds, learning_rates=eta_list)

# get preds
print('get preds')
train_preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
print('Train score is:', eval_wrapper(train_preds, train['Response'])) 
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
train_preds = np.clip(train_preds, -0.99, 8.99)
test_preds = np.clip(test_preds, -0.99, 8.99)

# train offsets
print('offset train')
offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
data = np.vstack((train_preds, train_preds, train['Response'].values))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
for j in range(num_classes):
    train_offset = lambda x: -apply_offset(data, x, j)
    offsets[j] = fmin_powell(train_offset, offsets[j])
    
# apply offsets to test
print('offset test')
data = np.vstack((test_preds, test_preds, test['Response'].values))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]     

print('final preds')
final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

print('preds to csv')
preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('xgb_offset_submission_3.csv')