import torch
import torch.utils.data as Data
from torch import nn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import math
import copy
import numpy as np 
import pandas as pd 
import pyarrow.parquet as pq
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import os

print(os.listdir("../input"))

time_step = 100
groupby_size = 100
def create_data(time_step, groupby_size):
    seq_len = 800000
    n_groupby = seq_len//groupby_size
    n_features = n_groupby//time_step
    
    meta_train = pd.read_csv('../input/metadata_train.csv')
    phase_train = pd.get_dummies(meta_train.phase).values
    phase_train = phase_train.reshape(phase_train.shape[0], 1, phase_train.shape[1])
    phase_train = np.repeat(phase_train, time_step, axis = 1)
    
    iter_size = 50
    n_iter = len(meta_train)//iter_size + 1
    data_x = np.array([]).reshape(-1, time_step, n_features)
    for k in range(0, n_iter):
        print('k =', k)
        start = iter_size*k
        end = iter_size*k + iter_size      
        data_xt = pq.read_pandas('../input/train.parquet', columns = [str(i) for i in meta_train.signal_id[start:end]]).to_pandas()
        data_xt = data_xt.values.reshape(n_groupby, groupby_size, data_xt.shape[-1])
        data_xt = data_xt.std(axis = 1).reshape(time_step, n_features, data_xt.shape[-1])
        data_xt = np.moveaxis(data_xt, [0, 1, 2], [1, 2, 0])
        data_x = np.concatenate((data_x, data_xt), 0)
        
    data_x = torch.tensor(np.concatenate((data_x, phase_train), 2)).float()
    torch.save(data_x, 'data_{}t_{}g'.format(time_step, groupby_size))


def create_test_data(time_step, groupby_size):
    seq_len = 800000
    n_groupby = seq_len//groupby_size
    n_features = n_groupby//time_step
        
    meta_test = pd.read_csv('../input/metadata_test.csv')
    phase_test = pd.get_dummies(meta_test.phase).values
    phase_test = phase_test.reshape(phase_test.shape[0], 1, phase_test.shape[1])
    phase_test = np.repeat(phase_test, time_step, axis = 1)
    
    iter_size = 100
    n_iter = len(meta_test)//iter_size + 1
    data_x = np.array([]).reshape(-1, time_step, n_features)
    for k in range(0, n_iter):
        print('k =', k)
        start = iter_size*k
        end = iter_size*k + iter_size      
        data_xt = pq.read_pandas('../input/test.parquet', columns = [str(i) for i in meta_test.signal_id[start:end]]).to_pandas()
        data_xt = data_xt.values.reshape(n_groupby, groupby_size, data_xt.shape[-1])
        data_xt = data_xt.std(axis = 1).reshape(time_step, n_features, data_xt.shape[-1])
        data_xt = np.moveaxis(data_xt, [0, 1, 2], [1, 2, 0])
        data_x = np.concatenate((data_x, data_xt), 0)
        
    data_x = torch.tensor(np.concatenate((data_x, phase_test), 2)).float()
    torch.save(data_x, 'data_{}t_{}g_test'.format(time_step, groupby_size))
    
##############################################################################################################
create_data(time_step = 16000, groupby_size = 50)
create_test_data(time_step = 16000, groupby_size = 50)
#create_data(time_step = 100, groupby_size = 100)
#create_data(time_step = 200, groupby_size = 100)
#create_data(time_step = 100, groupby_size = 50)
#create_data(time_step = 200, groupby_size = 50)
#create_data(time_step = 400, groupby_size = 50)
#create_data(time_step = 50, groupby_size = 50)