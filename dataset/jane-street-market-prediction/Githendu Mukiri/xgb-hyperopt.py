# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import xgboost as xgb
        
import warnings
warnings.filterwarnings("ignore")

# initialize the environment
import janestreet
env = janestreet.make_env() 
iter_test = env.iter_test()



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('../input/jane-street-market-prediction/train.csv')

train = train[train['weight'] != 0] #do not train data with 0 weight
train['action'] = (train['resp'].values > 0).astype('int') #only work with positive return expected

X_train = train.loc[:, train.columns.str.contains('feature')]
f_mean = X_train.mean()
X_train.fillna(f_mean) #fill na values with feature mean

y_train = train.loc[:, 'action']


print('Creating classifier...')
#create model

clf = xgb.XGBClassifier(
    
    n_estimators=31,
    max_depth=12,
    min_child_weight=10,
    gamma=0.7, 
    learning_rate=.025,
    missing=None,
    random_state=42,
    tree_method='gpu_hist',
    subsample=0.93,
    colsample_bytree=0.65,
)
print('Finished.')


print('Training classifier...')
#trainmodel
clf.fit(X_train, y_train)
print('Finished.')


for (test_df, sample_prediction_df) in iter_test:
    X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
    y_preds = clf.predict(X_test)
    sample_prediction_df.action = y_preds
    env.predict(sample_prediction_df)