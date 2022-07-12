# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from catboost import CatBoostClassifier, Pool

from tqdm import tqdm
import gc

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train.csv', na_values="-1")
test_df = pd.read_csv('../input/test.csv', na_values="-1")
id_test = test_df['id'].values

train_df = train_df.fillna(999)
test_df = test_df.fillna(999)



col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
train_df = train_df.drop(col_to_drop, axis=1)  
test_df = test_df.drop(col_to_drop, axis=1)  

for c in train_df.select_dtypes(include=['float64']).columns:
    train_df[c]=train_df[c].astype(np.float32)
    test_df[c]=test_df[c].astype(np.float32)
for c in train_df.select_dtypes(include=['int64']).columns[2:]:
    train_df[c]=train_df[c].astype(np.int8)
    test_df[c]=test_df[c].astype(np.int8)  


y_train = train_df['target'].values

train_features = [
    "ps_car_13",             
	"ps_reg_03",         
	"ps_ind_05_cat", 
	"ps_ind_03", 
	"ps_ind_15", 
	"ps_reg_02", 
	"ps_car_14", 
	"ps_car_12", 
	"ps_car_01_cat",  
	"ps_car_07_cat", 
	"ps_ind_17_bin", 
	"ps_car_03_cat", 
	"ps_reg_01", 
	"ps_car_15", 
	"ps_ind_01",  
	"ps_ind_16_bin", 
	"ps_ind_07_bin",  
	"ps_car_06_cat", 
	"ps_car_04_cat",  
	"ps_ind_06_bin", 
	"ps_car_09_cat",  
	"ps_car_02_cat",  
	"ps_ind_02_cat", 
	"ps_car_11",
	"ps_car_05_cat",  
	"ps_ind_08_bin",  
	"ps_car_08_cat", 
	"ps_ind_09_bin",  
	"ps_ind_04_cat",  
	"ps_ind_18_bin",
	"ps_ind_12_bin",
	"ps_ind_14", 
]

x_train = train_df[train_features]
x_test = test_df[train_features]
train_data = Pool(x_train, y_train)
test_data = Pool(x_test)
del train_df, test_df
gc.collect()

print('Starting the loop...')
num_ensembles = 6
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    model = CatBoostClassifier(random_seed = i+200, gradient_iterations = i+1 ,leaf_estimation_method ='Newton', learning_rate=0.057, l2_leaf_reg = 23, depth=6, od_pval=0.0000001, iterations = 877, loss_function='Logloss')
    fit_model = model.fit(train_data)
    y_pred +=  fit_model.predict_proba(test_data)[:,1]
y_pred /= num_ensembles
gc.collect()




# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('cat_predicts.csv', index=False)