### I want to dedicate myself to competitions, so I will be very happy if you give me some feedback

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import model_selection

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold,StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import lightgbm

from contextlib import contextmanager
import time
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
    
def read_df(percent=None):
    print("Importing datasets...")
    # Read data and merge
    df = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')
    
    print("Train shape: {}, Total fullVisitorId uniques: {}".format(df.shape, len(df['id'])))
    print("Test shape: {}, Total id uniques: {}".format(df_test.shape, len(df_test['id'])))
    
    df_test['scalar_coupling_constant'] = np.nan
    
    df = pd.concat([df, df_test])
    
    del df_test
    
    print("Full dataset shape: {}, Total id uniques: {}".format(df.shape, len(df['id'])))

    return df


def map_atom_info(df, atom_idx):

    df_structures = pd.read_csv('../input/structures.csv')
    df = pd.merge(df, df_structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    del df_structures
    return df
    
def dummies(df, list_cols):
    for col in list_cols:
        df_dummies = pd.get_dummies(df[col], drop_first=True, 
                                    prefix=(str(col)))
        df = pd.concat([df, df_dummies], axis=1)
        
    return df

def distance_atoms_map(df):
    print("Maping Atoms and Calculating the distance...")    
    df = map_atom_info(df, 0)
    df = map_atom_info(df, 1)
    
    train_p_0 = df[['x_0', 'y_0', 'z_0']].values
    train_p_1 = df[['x_1', 'y_1', 'z_1']].values

    df['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    df['dist_x'] = (df['x_0'] - df['x_1']) ** 2
    df['dist_y'] = (df['y_0'] - df['y_1']) ** 2
    df['dist_z'] = (df['z_0'] - df['z_1']) ** 2

    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df

def feature_engineering(df):
    print("Starting Feature Engineering...")
    df['type_0'] = df['type'].apply(lambda x: x[0])
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

    df = reduce_mem_usage(df)
    
    return df
    
good_columns = ['molecule_atom_index_0_dist_min','molecule_atom_index_0_dist_max',
                'molecule_atom_index_1_dist_min','molecule_atom_index_0_dist_mean',
                'molecule_atom_index_0_dist_std','dist','molecule_atom_index_1_dist_std',
                'molecule_atom_index_1_dist_max','molecule_atom_index_1_dist_mean',
                'molecule_atom_index_0_dist_max_diff','molecule_atom_index_0_dist_max_div',
                'molecule_atom_index_0_dist_std_diff','molecule_atom_index_0_dist_std_div',
                'atom_0_couples_count','molecule_atom_index_0_dist_min_div',
                'molecule_atom_index_1_dist_std_diff','molecule_atom_index_0_dist_mean_div',
                'atom_1_couples_count','molecule_atom_index_0_dist_mean_diff',
                'molecule_couples','atom_index_1','molecule_dist_mean',
                'molecule_atom_index_1_dist_max_diff', 'molecule_atom_index_0_y_1_std',
                'molecule_atom_index_1_dist_mean_diff','molecule_atom_index_1_dist_std_div',
                'molecule_atom_index_1_dist_mean_div', 'molecule_atom_index_1_dist_min_diff',
                'molecule_atom_index_1_dist_min_div','molecule_atom_index_1_dist_max_div',
                'molecule_atom_index_0_z_1_std', 'y_0','molecule_type_dist_std_diff',
                'molecule_atom_1_dist_min_diff','molecule_atom_index_0_x_1_std','molecule_dist_min',
                'molecule_atom_index_0_dist_min_diff', 'molecule_atom_index_0_y_1_mean_diff',
                'molecule_type_dist_min','molecule_atom_1_dist_min_div','atom_index_0',
                'molecule_dist_max','molecule_atom_1_dist_std_diff', 'molecule_type_dist_max',
                'molecule_atom_index_0_y_1_max_diff','molecule_type_0_dist_std_diff',
                'molecule_type_dist_mean_diff','molecule_atom_1_dist_mean',
                'molecule_atom_index_0_y_1_mean_div','molecule_type_dist_mean_div','type']
                
                
def Preprocessing(df):
    print("Starting Preprocessing..")
    # Threshold for removing correlated variables
    threshold = 0.97
    
    # Absolute value correlation matrix
    corr_matrix = df[df['scalar_coupling_constant'].notnull()].corr().abs()
    
    # Getting the upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print('There are %d columns to remove.' % (len(to_drop)))
    df = df.drop(columns = to_drop)
    print('Shape after drop high correlated features: ', df.shape)
    
    for f in ['atom_index_0', 'atom_index_1', 'atom_1', 'type_0', 'type']:
        if f in good_columns:
            lbl = LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))
            
    
    df = dummies(df, ['type', 'atom_1'])
    
    return df


def kfold_lightgbm(df, debug= False):
    print("Preparing the datasets...")
    # Divide in training/validation and test data
    train_df = df[df['scalar_coupling_constant'].notnull()].copy()
    test_df = df[df['scalar_coupling_constant'].isnull()].copy()
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    train_id = train_df['id'].copy()
    test_id = test_df['id'].copy()
    
    del df
    gc.collect()
    
    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(train_df.drop('scalar_coupling_constant', axis=1), 
                                                      train_df['scalar_coupling_constant'], 
                                                      test_size = 0.10, random_state = 21)
    
    X_train = X_train.drop(['id', 'atom_0', 'type', 'atom_1','molecule_name'], axis=1).values
    y_train = y_train.values
    X_val = X_val.drop(['id', 'atom_0', 'type', 'atom_1','molecule_name'], axis=1).values
    y_val = y_val.values
 
    print("Datasets: Prepared.")
    print("Training set has {} shape.".format(X_train.shape))
    print("Validation set has {} shape.".format(X_val.shape))

    params = {'boosting': 'gbdt', 'colsample_bytree': 1, 
              'learning_rate': 0.1, 'max_depth': 45, 'metric': 'mae',
              'min_child_samples': 50, 'num_leaves': 500, 
              'objective': 'regression', 'reg_alpha': 0.5, 
              'reg_lambda': 0.82, 'subsample': 0.5 }
              
    print("Starting the model...")    
    lgtrain = lightgbm.Dataset(X_train, label=y_train)
    lgval = lightgbm.Dataset(X_val, label=y_val)
 
    model_lgb = lightgbm.train(params, lgtrain, 10000, 
                          valid_sets=[lgtrain, lgval], early_stopping_rounds=250, 
                          verbose_eval=500)
                          
    print("Training: Done")
    
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    print("Preparing df_test to prediction...")
    X_test = test_df.drop(['id', 'atom_0', 'type', 'atom_1','molecule_name', 'scalar_coupling_constant'], axis=1).values
    
    print("Final test set has {} shape.".format(X_test.shape))
    print("Starting Prediction... ")
    pred = model_lgb.predict(X_test)
    print("Prediction: Done")
    
    del model_lgb, X_test
    gc.collect()
    
    print("Model finished.")
        
    print("Creating submission CSV File...")
    # Write submission file and plot feature importance
    if not debug:
        molecular_coupling = "molecular_coup_sub.csv"
        test_df['id'] = test_id
        test_df['scalar_coupling_constant'] = pred
        test_df[['id', 'scalar_coupling_constant']].to_csv(molecular_coupling, index= False)
        
    print("CSV Created: Done")    
    
    return test_df[['id', 'scalar_coupling_constant']].head()
 

def main(debug = False):
    p = 0.01 if debug else 1
    df = []
    
    with timer("Importing Datasets: "):    
        print("Importing datasets")
        df = read_df(df)
        gc.collect();

    with timer("Calculating Distance of Atoms: "):
        df = distance_atoms_map(df)
        print("Data Shape after Distance of Atoms", df.shape)
        gc.collect();

    with timer("Feature Engineering: "):
        df = feature_engineering(df)
        print("Data Shape after Feature Engineering: ", df.shape)
        gc.collect();

    with timer("Feature Preprocessing: "):
        df = Preprocessing(df)
        print("Shape After Feature Preprocessing: ", df.shape)
        gc.collect()

    with timer("Run LightGBM with cross validation"):
        model_prediction = kfold_lightgbm(df, debug= debug)

if __name__ == "__main__":
    with timer("Full model run "):
        df = main()
