"""
fork from https://www.kaggle.com/inversion/atomic-distance-benchmark
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import paired_euclidean_distances

from lightgbm import LGBMRegressor



# Map the atom structure data into train and test files
def map_atom_info(df, df_merge, atom_idx):
    cols_orig = df.columns.tolist()
    
    index_name = df.index.name
    index_series = df.index
    
    df = pd.merge(df, df_merge, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name',  'atom_index'])
    
    df.index.name = index_name
    df.index = index_series
    df = df.drop('atom_index', axis=1)
    cols_rename = {k: f'{k}_{atom_idx}' for k in df.columns.difference(cols_orig)}
    df = df.rename(columns=cols_rename)
    return df


def process_data_structure(df, df_merge):
    
    cols_origin = df.columns.tolist()

    df = map_atom_info(df, df_merge, 0)
    df = map_atom_info(df, df_merge, 1)

    #df['euclidean_distance'] = paired_euclidean_distances(
    #    df[['x_0', 'y_0', 'z_0']].values, df[['x_1', 'y_1', 'z_1']].values)
    
    df_p_0 = df[['x_0', 'y_0', 'z_0']].values
    df_p_1 = df[['x_1', 'y_1', 'z_1']].values
    df['euclidean_distance'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)
    
    # one group
    cols = ['euclidean_distance']
    groups = ['molecule_name', 'type']
    for group in groups:
        for col in cols:
            gp = df.groupby(group)[col]
            df[f'{col}_{group}_mean'] = gp.transform('mean')
            df[f'{col}_{group}_std'] = gp.transform('std')
            df[f'{col}_{group}_skew'] = gp.transform('skew')
    
            df[f'{col}_{group}_diff'] = df[col] - df[f'{col}_{group}_mean']
            df[f'{col}_{group}_zscore'] = df[f'{col}_{group}_diff'] / df[f'{col}_{group}_std']

    cols = df.columns.difference(cols_origin).tolist()    

    print(f'add {len(cols)}: {cols}:\n{df[cols].describe().T}')
    return df[cols].astype(np.float32)


def process_data_mst(df, df_merge):

    cols_origin = df.columns.tolist()

    df = map_atom_info(df, df_merge, 0)
    df = map_atom_info(df, df_merge, 1)

    cols = df.columns.difference(cols_origin)
    print(f'add {len(cols)}: {cols}')
    return df[cols].astype(np.float32)
    
    
def process_data_mc(df, df_merge):

    cols_origin = df.columns.tolist()

    df = map_atom_info(df, df_merge, 0)
    df = map_atom_info(df, df_merge, 1)

    cols = df.columns.difference(cols_origin)
    print(f'add {len(cols)}: {cols}')
    return df[cols].astype(np.float32)
    

def process_data_scc(df, df_merge):
    cols_origin = df.columns.tolist()
    df = pd.merge(
        df, df_merge, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1'],)
        
    cols = df.columns.difference(cols_origin)
    
    #for col in cols:
    #    gp = df.groupby('molecule_name')[col]
    #    df[f'{col}_mean'] = gp.transform('mean')
    #    df[f'{col}_std'] = gp.transform('std')
    #    df[f'{col}_skew'] = gp.transform('skew')
    #    
    #    df[f'{col}_diff'] = df[col] - df[f'{col}_mean']
    #    df[f'{col}_zscore'] = df[f'{col}_diff'] / df[f'{col}_std']
    
    #cols = df.columns.difference(cols_origin)
    
    print(f'add {len(cols)}: {cols}')
    return df[cols].astype(np.float32)


def cross_validation(params, train, test, n_splits=3):
    
    print(train.shape, test.shape)
    test = test.drop('molecule_name', axis=1)
    molecules = train.pop('molecule_name')
    y = train.pop('scalar_coupling_constant')
    
    yoof = np.zeros(len(train))
    yhat = np.zeros(len(test))
    gkf = GroupKFold(n_splits=n_splits) # we're going to split folds by molecules
    for fold, (in_index, oof_index) in enumerate(gkf.split(train, y, groups=molecules), 1):
        print(f'fold {fold} of {n_splits}')
        X_in, X_oof = train.iloc[in_index], train.iloc[oof_index]
        y_in, y_oof = y.iloc[in_index], y.iloc[oof_index]
        
        reg = LGBMRegressor(**params)
        reg.fit(X_in, y_in,
                eval_set=[(X_oof, y_oof)],
                verbose=100, eval_metric='mae',
                early_stopping_rounds=100)

        yoof[oof_index] = reg.predict(X_oof)
        yhat += reg.predict(test)

    yhat /= n_splits

    sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='id')
    benchmark = sample_submission.copy()
    benchmark['scalar_coupling_constant'] = yhat
    benchmark.to_csv('atomic_distance_benchmark.csv')


def process_data():
    train = pd.read_csv('../input/train.csv', index_col='id')
    test = pd.read_csv('../input/test.csv', index_col='id')

    structures = pd.read_csv('../input/structures.csv',)
    structures['atom'], _ = pd.factorize(structures['atom'])
    mst = pd.read_csv('../input/magnetic_shielding_tensors.csv',)
    mc = pd.read_csv('../input/mulliken_charges.csv')
    #scc = pd.read_csv('../input/scalar_coupling_contributions.csv')
    #del scc['type']

    train_df = list()    
    train_df.append(process_data_structure(train, structures))
    #train_df.append(process_data_mst(train, mst))
    #train_df.append(process_data_mc(train, mc))
    #train_df.append(process_data_scc(train, scc))
    train = train.join(train_df)

    test_df = list()    
    test_df.append(process_data_structure(test, structures))
    #test_df.append(process_data_mst(test, mst))
    #test_df.append(process_data_mc(test, mc))
    #test_df.append(process_data_scc(test, scc))    
    test = test.join(test_df)
    
    # Label Encoding
    for f in ['molecule_name', 'type', 'atom_index_0', 'atom_index_1']:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
        if f  == 'molecule_name':
            continue
        
        c = train[f].value_counts(normalize=False)
        train[f'{f}_cnt'] = train[f].map(c)
        test[f'{f}_cnt'] = test[f].map(c)
    
    return train, test

    
def main():

    params = {
        'boost_from_average': True,
        'is_provide_training_metric': True,
        'device': 'cpu',
        'objective': 'regression_l2',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'max_depth': 6,
        'n_estimators': 10000,
        'subsample_freq': 2,
        'metric_freq': 100,
        'verbosity': -1,
        'metric': 'mae',
        'colsample_bytree': 0.7,
        'learning_rate': 0.1,
        'min_child_samples': 100,
        'min_child_weight': 100.0,
        'min_split_gain': 0.1,
        'num_leaves': 11,
        'reg_alpha': 0.1,
        'reg_lambda': 0.005,
        'subsample': 0.75}
    
    train, test = process_data()
    cross_validation(params, train, test)
    

main()
