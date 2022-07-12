import os
import csv
from collections import defaultdict

import pandas as pd
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split

#PATH = os.system("ls ../input")
#os.system("echo \n\n")
#os.system("head ../input/*")

NUM_FACTS = 50
NUM_EPCS = 2
ALPHA = 2
LBDA = 0.015

def train_model_mf(train_set):
    
    # parse sparse matrix to colunar matrix
    train_set = train_set.tocsr()
    
    temp_idc_m = train_set.copy()
    nn_elems = int(train_set.indptr[-1])
    print("N elementos: ", nn_elems)
    temp_idc_m.data = np.arange(nn_elems)
    print("shape tem colviewm: ", temp_idc_m.data.shape)
    col_view_matrix = temp_idc_m.tocsc()  
    
    n_usr, n_cps = train_set.shape
    print("shape colunar: ", n_usr, n_cps)
    
    U = np.empty((n_usr, NUM_FACTS))
    C = NUM_FACTS**-0.5*np.random.random_sample((n_cps, NUM_FACTS))
    
    for epc in range(NUM_EPCS):
        
        print("Epoch: ", epc)
        
        # tune user factors first
        C_tuner = C.T.dot(C)
        for usr in range(n_usr):
            
            idx = train_set[usr].nonzero()[1]
            
            # check if user is in training set
            if idx.size:
                # user update
                u_ix = C[idx, :]
                m_usr_updated = C_tuner + ALPHA * u_ix.T.dot(u_ix) + np.diag(LBDA * np.ones(NUM_FACTS))
                U[usr, :] = np.dot(np.linalg.inv(m_usr_updated), (1 + ALPHA) * u_ix.sum(axis=0))
            else:
                # user index not loaded in treining set, leave empty
                U[usr, :] = np.zeros(NUM_FACTS)
        
        # tune cupon factors
        U_tuner = U.T.dot(U)
        for cps in range(n_cps):
            
            col = col_view_matrix[:, cps].copy()
            col.data = train_set.data[col.data]
            
            idx = col.nonzero()[0]
            
            print("tune cupon, idxsize: ", idx.size)
            
            # check if cupon is in training set
            if idx.size:
                # cupon update
                c_ix = U[idx, :]
                m_cupon_updated = U_tuner + ALPHA * c_ix.T.dot(c_ix) + np.diag(LBDA * np.ones(NUM_FACTS))
                C[cps, :] = np.dot(np.linalg.inv(m_cupon_updated), (1 + ALPHA) * c_ix.sum(axis=0))
            else:
                # cupon not found, leave empty
                C[cps, :] = np.zeros(NUM_FACTS)
                
    return U, C
    

detais_path = "../input/coupon_detail_train.csv"


df = pd.read_csv(detais_path, usecols=['USER_ID_hash', 'COUPON_ID_hash', 'ITEM_COUNT'])

print(df.columns)

df.columns = ['count', 'user', 'cupon']

df['user'] = pd.Categorical.from_array(pd.Series(df['user'], index=df.index)).codes.astype('int')
df['cupon'] = pd.Categorical.from_array(pd.Series(df['cupon'], index=df.index)).codes.astype('int')
df = df.convert_objects(convert_numeric=True)

print(df.columns)

print(df.describe())

"""

train, validation = train_test_split(df, test_size=0.1)

train_sparse = csr_matrix(train.values)

print("START TRAINING")

user_features, cupon_features =  train_model_mf(train_sparse)

print("TREINOU!")

for i in range(0, 10):
    print(user_features.shape)
    print(cupon_features.shape)



"""
