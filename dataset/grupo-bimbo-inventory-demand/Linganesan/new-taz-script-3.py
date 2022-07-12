import numpy as np
import pandas as pd

from csv import DictReader
from math import sqrt, log, expm1


# Define size limits for each field to save memory
dtypes_test = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16}

# python 3.5 version: {**dtypes_test, **{'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32, 'Demanda_uni_equil': np.int16}}
dtypes_train = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16,
               'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32, 'Demanda_uni_equil': np.int16}

# Now load train+test data
print ('Reading train dataset')
df_train = pd.read_csv('../input/train.csv', dtype = dtypes_train, usecols = ['Producto_ID', 'Ruta_SAK', 'Agencia_ID', 'Cliente_ID', 'Demanda_uni_equil'])

print ('Reading test dataset')
df_test = pd.read_csv('../input/test.csv', dtype = dtypes_test)

df_train['log_demand'] = np.log1p(df_train.Demanda_uni_equil)
print(df_train['log_demand'].head())

demand_mean = df_train['log_demand'].mean()
print('demand mean:')
print(demand_mean)

# Compute the three log means - the processing code works faster (less slowly?) if they're converted to Dictionaries.
demand_prod = (df_train.groupby(['Producto_ID']))['log_demand'].mean().to_dict()
print('demand prod:')
print(demand_prod)
submission = np.zeros(len(df_test))

k = 0

# We need to handle one product at a time, otherwise the dictionaries get too large...

def process_product(product_id, sub):
    global k
    
    #Get subsets from train,test dataframes which contains same product ID 
    df_train_subset = df_train[df_train['Producto_ID'] == product_id]
    df_test_subset = df_test[df_test['Producto_ID'] == product_id]
    
    #Gather (product_ID,Route_ID)	
    demand_prod_ruta = (df_train_subset.groupby(['Producto_ID', 'Ruta_SAK']))['log_demand'].mean().to_dict()
    print('demand_prod_ruta:')
    # print( demand_prod_ruta.head())
    demand_prod_cli_age = (df_train_subset.groupby(['Producto_ID', 'Cliente_ID', 'Agencia_ID']))['log_demand'].mean().to_dict()
    print('demand_prod_cli_age:')
    # print( demand_prod_cli_age.head())
    #################### 
    df_test_p = df_test_subset['Producto_ID']
    print('df_test_p:')
    # print( df_test_p.head())
    df_test_pr = df_test_subset[['Producto_ID', 'Ruta_SAK']]
    print('df_test_pr:')
    # print( df_test_pr.head())
    df_test_l_pr = list(df_test_pr.itertuples(index=False, name=None))
    print('df_test_l_pr:')
    # print( df_test_l_pr.head())
    df_test_pca = df_test_subset[['Producto_ID', 'Cliente_ID', 'Agencia_ID']]
    print('df_test_pca:')
    # print( df_test_pca.head())
    df_test_l_pca = list(df_test_pca.itertuples(index=False, name=None))
    print('df_test_l_pca:')
    # print( df_test_l_pca.head())
    ####################    

    output = []

    # make a meta-tuple of each of the tuples used to do log-mean lookups
    for z in zip(df_test_subset.id, df_test_p, df_test_l_pr, df_test_l_pca):

        # Work in order of preference.  With straight dicts this is faster than try/except
        if z[3] in demand_prod_cli_age:
            o = (demand_prod_cli_age[z[3]] )
        elif z[2] in demand_prod_ruta:
            o = (demand_prod_ruta[z[2]] )
        elif z[1] in demand_prod:
            o = (demand_prod[z[1]]) 
        else:
            o = demand_mean

        sub[z[0]] = np.expm1(o) 

for p in df_test.Producto_ID.unique():
    process_product(p, submission)            

# Now output
df_test['Demanda_uni_equil'] = submission

df_submit = df_test[['id', 'Demanda_uni_equil']]
df_submit = df_submit.set_index('id')
df_submit.to_csv('taz_submission.csv')