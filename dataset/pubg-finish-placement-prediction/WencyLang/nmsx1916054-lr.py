# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import gc, sys
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
# Reduce the usage of memory
# Ref: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):
    '''
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    '''
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
	

def feature_engineering(is_train=True,debug=True):
    test_idx = None
    if is_train: 
        print("processing train.csv")
        if debug == True:
            df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv', nrows=10000)
        else:
            df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')           

        df = df[df['maxPlace'] > 1]
    else:
        print("processing test.csv")
        df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
        test_idx = df.Id
    
    #df = reduce_mem_usage(df)
    
    print("remove some columns")
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")  
    
    y = None
    
    
    if is_train: 
        print("get target")
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    else: df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = df_out
    
    feature_names = list(df_out.columns)

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y, feature_names, test_idx
#df_train = reduce_mem_usage(pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv'))
#df_test = reduce_mem_usage(pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv'))	
df_train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
df_test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
X_train, y_train, train_columns, _ = feature_engineering(True,False)
X_test, _, _ , test_idx = feature_engineering(False,True)
#X_train =reduce_mem_usage(X_train)
#X_test = reduce_mem_usage(X_test)
#X_train =reduce_mem_usage(X_train)
#X_test = reduce_mem_usage(X_test)

LR_model = LinearRegression(n_jobs=4, normalize=True)
LR_model.fit(X_train,y_train)
LR_model.score(X_train,y_train)

y_pred_train = LR_model.predict(X_train)
y_pred_test = LR_model.predict(X_test)

y_pred_train[y_pred_train>1] = 1
y_pred_train[y_pred_train<0] = 0

#f, ax = plt.subplots(figsize=(10,10))
#plt.scatter(y_train, y_pred_train)
#plt.xlabel("y")
#plt.ylabel("y_pred_train")
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.show()

y_pred_test[y_pred_test>1] = 1
y_pred_test[y_pred_test<0] = 0

df_test['winPlacePerc'] = y_pred_test
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)