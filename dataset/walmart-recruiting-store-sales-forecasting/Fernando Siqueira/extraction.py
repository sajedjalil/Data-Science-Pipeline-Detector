import pandas as pd
import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


TRAIN_DATASET = '/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip'
STORES_DATASET = '/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv'
FEATURES_DATASET = '/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip'
TEST_DATASET = '/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip'

def get_train_dataset():
    
    df_train = pd.read_csv(TRAIN_DATASET)
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    
    return df_train


def get_stores_dataset():
    
    return pd.read_csv(STORES_DATASET)


def get_features_dataset():
    
    df_features = pd.read_csv(FEATURES_DATASET)
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    return df_features


def get_test_dataset():
    
    df_test = pd.read_csv(TEST_DATASET)
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    
    return df_test
