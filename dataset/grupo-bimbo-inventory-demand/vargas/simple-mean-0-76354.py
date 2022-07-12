import pandas as pd
import numpy as np
import os

# Kaggle/Bimbo result 0.76354

# Define size limits for each field to save memory
dtypes_test = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16}
dtypes_train = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16,
               'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32, 'Demanda_uni_equil': np.int16}
               
# Load data
train = pd.read_csv('../input/train.csv',dtype = dtypes_train)
test = pd.read_csv('../input/test.csv',dtype = dtypes_test)



# Obtain a subset of data
train_sub = train[['Agencia_ID','Cliente_ID','Producto_ID','Demanda_uni_equil']]

# Group by component vars
train_groups = train_sub.groupby(['Agencia_ID','Cliente_ID','Producto_ID',],group_keys=True)

# Obtain the mean for each group
train_groups_means = train_groups.agg(np.mean)

# Ungroup & prepare shape for merging with test
pd.options.display.multi_sparse = False
test_sub = test[['Agencia_ID', 'Cliente_ID', 'Producto_ID']]
train_groups_means = train_groups_means.reset_index([0,1,2])

# Merge with test
result = pd.merge(test_sub, train_groups_means, how='left', on=['Agencia_ID', 'Cliente_ID', 'Producto_ID'], sort = False)

# Fill NaN and write to file for submission
result = result.fillna(1)
result.to_csv(path_or_buf='../input/submissions/BayesianRidge.csv',
    columns=['Demanda_uni_equil'],index_label='id')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.