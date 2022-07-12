import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)
    
def get_params():
    
    params = {}
    params['objective'] = 'reg:linear'     
    params['eta'] = 0.05
    params['min_child_weight'] = 240
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.67
    params['silent'] = 1
    params['max_depth'] = 6
    plst = list(params.items())

    return plst
    
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

# global variables
columns_to_drop = ['Id', 'Response', 'Medical_History_10', 'Medical_History_24',
                    'BMI', 'Ins_Age', 'Wt', 'Ht', 'Family_Hist_4', 'Family_Hist_2', ]
xgb_num_rounds = 700
num_classes = 8
eta_list = [0.05] * 200 + [0.02] * 500 + [0.01] * 500

print('Load the data using pandas')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

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

all_data.fillna(-1, inplace=True)

factor = 1.0

all_data['Ins_Age_Family_Hist_2']             = factor * all_data['Ins_Age']        / all_data['Family_Hist_2']
all_data['Family_Hist_4_Ins_Age']             = factor * all_data['Family_Hist_4']  / all_data['Ins_Age']
all_data['Family_Hist_4_Insurance_History_5'] = factor * all_data['Family_Hist_4']  / all_data['Insurance_History_5']
all_data['Ins_Age_BMI']                       = factor * all_data['Ins_Age']        / all_data['BMI']
all_data['Wt_Ins_Age']                        = factor * all_data['Wt']             / all_data['Ins_Age']
all_data['Family_Hist_4_Family_Hist_2']       = factor * all_data['Family_Hist_4']  / all_data['Family_Hist_2']
all_data['Family_Hist_4_Employment_Info_1']   = factor * all_data['Family_Hist_4']  / all_data['Employment_Info_1']
all_data['Family_Hist_4_BMI']                 = factor * all_data['Family_Hist_4']  / all_data['BMI']
all_data['BMI_Insurance_History_5']           = factor * all_data['BMI']            / all_data['Insurance_History_5']
all_data['Ht_Employment_Info_1']              = factor * all_data['Ht']             / all_data['Employment_Info_1']
all_data['BMI_Product_Info_4']                = factor * all_data['BMI']            / all_data['Product_Info_4']
all_data['BMI_Age']                           = factor * all_data['BMI']            * all_data['Ins_Age']
all_data['Product_Info_4_x_Family_Hist_2']    = factor * all_data['Product_Info_4'] * all_data['Family_Hist_2']
all_data['Ins_Age_x_Insurance_History_5']     = factor * all_data['Ins_Age']        * all_data['Insurance_History_5']
all_data['Ins_Age_x_Family_Hist_3']           = factor * all_data['Ins_Age']        * all_data['Family_Hist_3']

all_data.fillna(-1, inplace=True)

#all_data['Ins_Age_Family_Hist_2']             = all_data['Ins_Age_Family_Hist_2'].astype(int)
#all_data['Family_Hist_4_Ins_Age']             = all_data['Family_Hist_4_Ins_Age'].astype(int)
#all_data['Family_Hist_4_Insurance_History_5'] = all_data['Family_Hist_4_Insurance_History_5'].astype(int)
#all_data['Ins_Age_BMI']                       = all_data['Ins_Age_BMI'].astype(int)
#all_data['Wt_Ins_Age']                        = all_data['Wt_Ins_Age'].astype(int)
#all_data['Family_Hist_4_Family_Hist_2']       = all_data['Family_Hist_4_Family_Hist_2'].astype(int)
#all_data['Family_Hist_4_Employment_Info_1']   = all_data['Family_Hist_4_Employment_Info_1'].astype(int)
#all_data['Family_Hist_4_BMI']                 = all_data['Family_Hist_4_BMI'].astype(int)
#all_data['BMI_Insurance_History_5']           = all_data['BMI_Insurance_History_5'].astype(int)
#all_data['Ht_Employment_Info_1']              = all_data['Ht_Employment_Info_1'].astype(int)
#all_data['BMI_Product_Info_4']                = all_data['BMI_Product_Info_4'].astype(int)
#all_data['Product_Info_4_x_Family_Hist_2']    = all_data['Product_Info_4_x_Family_Hist_2'].astype(int)
#all_data['Ins_Age_x_Insurance_History_5']     = all_data['Ins_Age_x_Insurance_History_5'].astype(int)
#all_data['Ins_Age_x_Family_Hist_3']           = all_data['Ins_Age_x_Family_Hist_3'].astype(int)

med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

print('Eliminate missing values')    
all_data.fillna(-1, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# split train and test
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()

# convert data to xgb data structure
xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1))    

# train model
model = xgb.train(get_params(), xgtrain, xgb_num_rounds, learning_rates=eta_list) 

# get preds
train_preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
print('Train score is:', eval_wrapper(train_preds, train['Response'])) 
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
train_preds = np.clip(train_preds, -0.99, 8.99)
test_preds = np.clip(test_preds, -0.99, 8.99)

checkpoints = np.array([-100, 2.9843, 3.6675, 4.3929, 4.8428, 5.4960, 6.2341, 6.8287, 100])

def get_prediction_by_checkpoints(data, points):
    result = data.copy()
    for i in range(num_classes):
        result[np.where((data >= points[i]) & (data < points[i+1]))] = i+1
    result = result.astype(int)
    return result

def make_points(checkpoints, j, x):
    p = np.copy(checkpoints)
    p[j] = x
    return p

for step in range(2):    
    for j in range(1,8):
        train_chk = lambda x: -eval_wrapper(get_prediction_by_checkpoints(train_preds, make_points(checkpoints, j, x)), train['Response'])
        checkpoints[j] = fmin_powell(train_chk, checkpoints[j])        

print(checkpoints)
train_preds_adj = get_prediction_by_checkpoints(train_preds, checkpoints)
print('Train score is (adj): %.6f' % (eval_wrapper(train_preds_adj, train['Response'])))
    
final_test_preds = get_prediction_by_checkpoints(test_preds, checkpoints)

preds_out = pd.DataFrame({'Id': test['Id'].values, 'Response': final_test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('xgb_offset_submission_3.csv')