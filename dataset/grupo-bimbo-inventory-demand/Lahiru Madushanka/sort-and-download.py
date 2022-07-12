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
dtypes_test = {'Semana': np.int8, 'Canal_ID': np.int8, 'Producto_ID': np.uint16,'Ruta_SAK':np.int16,'Agencia_ID': np.int16}

# python 3.5 version: {**dtypes_test, **{'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32, 'Demanda_uni_equil': np.int16}}
dtypes_train = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16,
               'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32, 'Demanda_uni_equil': np.int16}

# Now load train+test data
datadir = '../input/'
df_train = pd.read_csv(datadir + 'train.csv', dtype = dtypes_train, usecols = ['Producto_ID', 'Ruta_SAK', 'Agencia_ID', 'Cliente_ID', 'Demanda_uni_equil','Canal_ID'])
df_test = pd.read_csv(datadir + 'test.csv', dtype = dtypes_test)

df_train['log_demand'] = np.log1p(df_train.Demanda_uni_equil)

# Compute the three log means - the processing code works faster (less slowly?) if they're converted to Dictionaries.
demand_prod = (df_train.groupby(['Producto_ID']))['log_demand'].median().to_dict()
#demand_prod_ruta = (df_train.groupby(['Producto_ID', 'Ruta_SAK']))['log_demand'].mean().to_dict()
#demand_prod_cli_age = (df_train.groupby(['Producto_ID', 'Cliente_ID', 'Agencia_ID']))['log_demand'].mean().to_dict()

submission = np.zeros(len(df_test))

k = 0
'''
Cliente_ID,Producto_ID,Agencia_ID,Ruta_SAK,Canal_ID [0.945129]
Cliente_ID,Producto_ID,Agencia_ID,Ruta_SAK [0.945122]
Cliente_ID,Producto_ID,Ruta_SAK,Canal_ID [0.933302]
Cliente_ID,Producto_ID,Ruta_SAK [0.933294]
Cliente_ID,Producto_ID,Agencia_ID,Canal_ID [0.933238]
Cliente_ID,Producto_ID,Agencia_ID [0.933196]
Cliente_ID,Producto_ID,Canal_ID [0.924643]
Cliente_ID,Producto_ID [0.923082]'''
# We need to handle one product at a time, otherwise the dictionaries get too large...

def process_product(product_id, sub):
    global k
    
    df_train_subset = df_train[df_train['Producto_ID'] == product_id]
    df_test_subset = df_test[df_test['Producto_ID'] == product_id]
    
    demand_prod_ruta = (df_train_subset.groupby(['Producto_ID', 'Cliente_ID']))['log_demand'].median().to_dict()
    demand_prod_cli_age = (df_train_subset.groupby(['Producto_ID', 'Cliente_ID','Agencia_ID']))['log_demand'].median().to_dict()
   
    
    df_test_p = df_test_subset['Producto_ID']

    df_test_pr = df_test_subset[['Producto_ID', 'Cliente_ID']]
    df_test_l_pr = list(df_test_pr.itertuples(index=False, name=None))

    df_test_pca = df_test_subset[['Producto_ID', 'Cliente_ID', 'Agencia_ID']]
    df_test_l_pca = list(df_test_pca.itertuples(index=False, name=None))
    
    df_test_pcaRC = df_test_subset[['Producto_ID', 'Cliente_ID', 'Agencia_ID','Ruta_SAK','Canal_ID']]
    df_test_l_pcaRC = list(df_test_pcaRC.itertuples(index=False, name=None))
    

    output = []

    # make a meta-tuple of each of the tuples used to do log-mean lookups
    for z in zip(df_test_subset.id, df_test_p, df_test_l_pr, df_test_l_pca,df_test_pcaRC):

        if z[4] in demand_prod_cli_age:
            o = demand_prod_cli_age[z[4]]
        elif z[3] in demand_prod_cli_age:
            o = demand_prod_cli_age[z[3]]
        elif z[2] in demand_prod_ruta:
            o = demand_prod_ruta[z[2]]
        elif z[1] in demand_prod:
            o = demand_prod[z[1]]
        else:
            o = 2
            
        sub[z[0]] = np.expm1(o) * .9

for p in df_test.Producto_ID.unique():
    process_product(p, submission)            

# Now output
df_test['Demanda_uni_equil'] = submission

df_submit = df_test[['id', 'Demanda_uni_equil']]
df_submit = df_submit.set_index('id')
df_submit.to_csv('meantest2a.csv')