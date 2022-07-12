import numpy as np 
import pandas as pd 
from catboost import Pool, CatBoostRegressor
from datetime import datetime as dt
from sklearn.model_selection import train_test_split


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ss = pd.read_csv('../input/sample_submission.csv')

DATE_TODAY = dt(2019, 1, 26)


def one_hot_encoder(df, nan_as_category=True):
    original_columns = df.columns.tolist()

    categorical_columns = list(filter(lambda c: c in ['object'], df.dtypes))
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)

    new_columns = list(filter(lambda c: c not in original_columns, df.columns))
    return df, new_columns

def feature_engineering(df):
    df['first_active_month'] = pd.to_datetime(df['first_active_month'], errors='coerce')
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month    
    df['hour'] = df['first_active_month'].dt.hour
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['weekday'] = df['first_active_month'].dt.weekday
    df['weekend'] = (df['first_active_month'].dt.weekday >= 5).astype(int)
    
    df['elapsed_time'] = (DATE_TODAY - df['first_active_month']).dt.days
    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    for f in feature_cols:    
        df['days_' + f] = df['elapsed_time'] * df[f]
        df['days_' + f + '_ratio'] = df[f] / df['elapsed_time']


feature_engineering(test)
feature_engineering(train)

X_train, X_test, y_train, y_test = train_test_split(train.drop(['first_active_month', 'card_id', 'target'],1), train.target, test_size=0.23, random_state=23)

train_pool = Pool(X_train,
                  label=y_train,  
                  cat_features=[0,1,2])
                  
val_pool = Pool(X_test,
                  label=y_test,  
                  cat_features=[0,1,2])

test_pool = Pool(test.drop(['first_active_month', 'card_id',],1), cat_features=[0,1,2])

params = {'od_pval': 0.0001, 'random_seed':1111, 'depth':3, 'learning_rate':0.01, 'iterations':2000, 'random_seed':1, 'od_wait':20,
        'loss_function': 'RMSE'}
        
n_runs = 10
y_pred = 0.0
for i in range(n_runs):
    params['random_seed'] + 1*100
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    y_pred +=  model.predict(test_pool)

y_pred /= n_runs

ss['target'] = y_pred
ss.to_csv('sub.csv', index=False)
