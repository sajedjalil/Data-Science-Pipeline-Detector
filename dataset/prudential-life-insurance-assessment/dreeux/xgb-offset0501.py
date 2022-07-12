import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa



# global variables
columns_to_drop = ['Id', 'Response']
xgb_num_rounds = 500
num_classes = 8

print("Load the data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# combine train and test
all_data = train.append(test)

# create any new variables    
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[1]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[2]

all_data.to_csv('xgb_offset_submission0501.csv')


 