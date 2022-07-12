# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pickle

import numpy as np
import pandas as pd

from csv import DictReader
from math import sqrt, log, expm1
from datetime import datetime

'''
min/max for training data (test data is within the same bounds bit-wise)

Semana 3 9
Agencia_ID 1110 25759
Canal_ID 1 11
Ruta_SAK 1 9991
Cliente_ID 26 2015152015
Producto_ID 41 49997
Venta_uni_hoy 0 7200
Venta_hoy 0.0 647360.0
Dev_uni_proxima 0 250000
Dev_proxima 0.0 130760.0
Demanda_uni_equil 0 5000
'''

# Define size limits for each field to save memory
dtypes_test = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16}

# python 3.5 version: {**dtypes_test, **{'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32, 'Demanda_uni_equil': np.int16}}
dtypes_train = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16,
               'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32, 'Demanda_uni_equil': np.int16}

# Now load train+test data
datadir = '../input/'
df_train = pd.read_csv(datadir + 'train.csv', dtype = dtypes_train, usecols = ['Producto_ID', 'Ruta_SAK', 'Agencia_ID', 'Cliente_ID', 'Demanda_uni_equil'])
df_test = pd.read_csv(datadir + 'test.csv', dtype = dtypes_test)

df_train['log_demand'] = np.log1p(df_train.Demanda_uni_equil)

demand_mean = df_train['log_demand'].mean()

# Compute the three log means - the processing code works faster (less slowly?) if they're converted to Dictionaries.
demand_prod = (df_train.groupby(['Producto_ID']))['log_demand'].mean().to_dict()
#demand_prod_ruta = (df_train.groupby(['Producto_ID', 'Ruta_SAK']))['log_demand'].mean().to_dict()
#demand_prod_cli_age = (df_train.groupby(['Producto_ID', 'Cliente_ID', 'Agencia_ID']))['log_demand'].mean().to_dict()

submission = np.zeros(len(df_test))

k = 0

# We need to handle one product at a time, otherwise the dictionaries get too large...

def process_product(product_id, sub):
    global k
    
    df_train_subset = df_train[df_train['Producto_ID'] == product_id]
    df_test_subset = df_test[df_test['Producto_ID'] == product_id]
    
    demand_prod_ruta = (df_train_subset.groupby(['Producto_ID', 'Ruta_SAK']))['log_demand'].mean().to_dict()
    demand_prod_cli_age = (df_train_subset.groupby(['Producto_ID', 'Cliente_ID', 'Agencia_ID']))['log_demand'].mean().to_dict()
    
    df_test_p = df_test_subset['Producto_ID']

    df_test_pr = df_test_subset[['Producto_ID', 'Ruta_SAK']]
    df_test_l_pr = list(df_test_pr.itertuples(index=False, name=None))

    df_test_pca = df_test_subset[['Producto_ID', 'Cliente_ID', 'Agencia_ID']]
    df_test_l_pca = list(df_test_pca.itertuples(index=False, name=None))
    
    output = []

    # make a meta-tuple of each of the tuples used to do log-mean lookups
    for z in zip(df_test_subset.id, df_test_p, df_test_l_pr, df_test_l_pca):

        # Work in order of preference.  With straight dicts this is faster than try/except
        if z[3] in demand_prod_cli_age:
            o = (demand_prod_cli_age[z[3]] * .917) + .1
        elif z[2] in demand_prod_ruta:
            o = (demand_prod_ruta[z[2]] * .85) + .1
        elif z[1] in demand_prod:
            o = (demand_prod[z[1]] * .97) + .15
        else:
            o = demand_mean * 1.4

        sub[z[0]] = np.expm1(o) 

for p in df_test.Producto_ID.unique():
    process_product(p, submission)            

# Now output
df_test['Demanda_uni_equil'] = submission

df_submit = df_test[['id', 'Demanda_uni_equil']]
df_submit = df_submit.set_index('id')
df_submit.to_csv('meantest3.csv')