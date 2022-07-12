#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as systime
import datetime as dtime
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import gc


month_enum={1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'}


TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'

train_raw = pd.read_csv('../input/'+TRAIN_FILENAME, parse_dates=['Dates'], index_col=False)
test_raw = pd.read_csv('../input/'+TEST_FILENAME, parse_dates=['Dates'], index_col=False)

def feature_engineering(data):
    
    #Get binarized weekdays, districts, and hours.
    days = pd.get_dummies(data.DayOfWeek)
    district = pd.get_dummies(data.PdDistrict)
    month = pd.get_dummies(data.Dates.dt.month.map(month_enum))
    hour = data.Dates.dt.hour
    hour = pd.get_dummies(hour)  
    #Append newly created dummy variables to dataframe
    new_data = pd.concat([hour, month, days, district], axis=1)
    
    return new_data
    

train = feature_engineering(train_raw)
#test = pd.concat([test_raw['Id'],feature_engineering(test_raw)], axis=1)
test = feature_engineering(test_raw)

cat_enc = LabelEncoder()
cat_enc.fit(train_raw['Category'])
train['CategoryEncoded'] = cat_enc.transform(train_raw['Category'])

x_cols = list(train.columns[0:53].values)


# Set parameters for XGBoost
def set_param():
    
    # setup parameters for xgboost
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.4
    param['silent'] = 0
    param['nthread'] = 4
    param['num_class'] = num_class
    param['eval_metric'] = 'mlogloss'

    # Model complexity
    param['max_depth'] = 8 #set to 8
    param['min_child_weight'] = 1
    param['gamma'] = 0 
    param['reg_alfa'] = 0.05

    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8 #set to 1

    # Imbalanced data
    param['max_delta_step'] = 1
    
    return param
    

# Split into train/validate test
train_data, validate_data = train_test_split(train, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(train_data[x_cols], label=train_data['CategoryEncoded'])
dtest = xgb.DMatrix(validate_data[x_cols], label=validate_data['CategoryEncoded'])

num_class = train['CategoryEncoded'].nunique()

param = set_param()
watchlist = [ (dtrain,'train'), (dtest, 'eval') ]
num_round = 12

# Train XGBoost    
bst = xgb.train(param, dtrain, num_round, watchlist);
yprob = bst.predict(dtest).reshape( validate_data['CategoryEncoded'].shape[0], num_class)
ylabel = np.argmax(yprob, axis=1)

# predict with test data
dtest = xgb.DMatrix(test[x_cols])
predicted = bst.predict(dtest)

# Make the output data frame by mapping the probability estimates to categories
#crime = cat_enc.fit_transform(train_raw.Category)
result=pd.DataFrame(predicted, columns=cat_enc.classes_)
submission=result.round(2)
# Appending the Index column
#submission= pd.concat([test_raw['Id'], result], axis=1)

del train_raw
del test_raw
del train
del test
del train_data
del dtrain
del dtest

gc.collect()

submission.to_csv('submit.csv', index = True, index_label = 'Id')

