# Script generates csv fils with training and test sets
# Those sets are created based on original competition data + feature engeenering 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
from time import time

def main():
    print("Feature generation...")
    train, test = get_train_and_test_sets()
    
    print("Encoding train and test sets")
    encoder = LabelEncoder()
    encoder.fit(structures_df.atom.values)
    
    train = encode_categorical_features(train, encoder)
    test = encode_categorical_features(test, encoder)
    
    print("Saving files...")
    train.to_pickle('train_set.zip')
    test.to_pickle('test_set.zip')
    
    print("Done")

def get_train_and_test_sets():
    train_batches = split_set_to_batches(train_df, "train")
    test_batches = split_set_to_batches(test_df, "test")
    
    all_batches = train_batches + test_batches
    
    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(create_features, all_batches)
    
    # Sort all results by batch id to recreate original order
    results = sorted(results, key=lambda batch: batch[2])
    
    train_results = [df for source_name, df, _ in results if source_name == "train"]
    train_set = pd.concat(train_results, ignore_index=True)
    
    test_results = [df for source_name, df, _ in results if source_name == "test"]
    test_set = pd.concat(test_results, ignore_index=True)
    
    return train_set, test_set

def encode_categorical_features(data, encoder):
    data['atom_0'] = encoder.transform(data['atom_0'].values)
    data['atom_1'] = encoder.transform(data['atom_1'].values)
    
    return data

def split_set_to_batches(original_data, source_name):
    BATCH_SIZE = 1000000
    start_idx = 0
    end_idx = BATCH_SIZE
    batches = list()
    index = 0
    
    while (end_idx < len(original_data)):
        batch = original_data.iloc[start_idx : end_idx]
        
        batches.append((batch, source_name, index))
        start_idx += BATCH_SIZE
        end_idx += BATCH_SIZE
        index += 1
        
    # add last, smaller batch
    batch = original_data.iloc[start_idx:]
    batches.append((batch, source_name, index))
    
    return batches

# ----------------------------- Helper functions -------------------------------------------------

def create_features(batch):
    data, source_name, index = batch
    data_with_atoms_info = add_atoms_info(data)
    
    data_with_features = add_distances(data_with_atoms_info)
    data_with_features = add_bounds_count(data_with_features)
    data_with_features = drop_not_important_columns(data_with_features)
    
    return source_name, data_with_features, index

def drop_not_important_columns(data):
    columns_to_drop = [
        'molecule_name',
        'atom_index_0',
        'atom_index_1',
        'x_0', 'y_0', 'z_0',
        'x_1', 'y_1', 'z_1',
    ]
    
    return data.drop(columns_to_drop, axis=1)

def add_bounds_count(data):
    data['bounds_count'] = data.type.apply(lambda x: int(x[0]))
    
    return data

def add_distances(data):
    data['dist_x'] = (data['x_0'] - data['x_1']).abs()
    data['dist_y'] = (data['y_0'] - data['y_1']).abs()
    data['dist_z'] = (data['z_0'] - data['z_1']).abs()
    
    point_0 = data[['x_0', 'y_0', 'z_0']].values
    point_1 = data[['x_1', 'y_1', 'z_1']].values
    
    data['distance'] = np.linalg.norm(point_0 - point_1, axis=1)
    
    return data

def add_atoms_info(data_set):    
    for atom_idx in range(2):
        data_set = pd.merge(data_set, structures_df, how = 'left',
                      left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                      right_on = ['molecule_name',  'atom_index'])
    
        data_set = data_set.drop('atom_index', axis=1)
        data_set = data_set.rename(columns={'atom': f'atom_{atom_idx}',
                                'x': f'x_{atom_idx}',
                                'y': f'y_{atom_idx}',
                                'z': f'z_{atom_idx}'})
    return data_set     


print("Loading data")
train_df = pd.read_csv("../input/train.csv")
train_df = train_df.drop(['scalar_coupling_constant', 'id'], axis=1)
test_df = pd.read_csv("../input/test.csv")
structures_df = pd.read_csv("../input/structures.csv")
main()