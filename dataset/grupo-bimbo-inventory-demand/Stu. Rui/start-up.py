# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
import math

# Product Mapping ID
producto_tabla = pd.read_csv('../input/producto_tabla.csv')
producto_tabla['First'] = producto_tabla['NombreProducto'].apply(lambda x: x.split(' ')[0])
producto_dict = dict()
for (i,x) in enumerate(pd.Series(producto_tabla['First'].unique())):
    producto_dict[x] = i
producto_tabla['MapFirst'] = producto_tabla['First'].apply(lambda x: producto_dict[x])
temp = producto_tabla[['Producto_ID', 'MapFirst']].values
producto_cluster = dict()
for i in range(temp.shape[0]):
    producto_cluster[temp[i,0]] = temp[i,1]

# Town State Mapping ID
town_state = pd.read_csv('../input/town_state.csv')
town_state['T_S'] = town_state['Town'] + ':' + town_state['State']
town_state_dict = dict()
s_t = town_state['T_S'].unique()
for i in range(s_t.shape[0]):
    ts = s_t[i].split(':')
    town_state_dict[(ts[0], ts[1])] = i
temp = town_state[['Agencia_ID', 'Town', 'State']].values
town_state_cluster = dict()
for i in range(temp.shape[0]):
    if town_state_cluster.get(temp[i,0], -1) == -1:
        town_state_cluster[temp[i,0]] = town_state_dict[(temp[i,1], temp[i,2])]
    else:
        pass
    
# Cliente ID Map
cliente_tabla = pd.read_csv('../input/cliente_tabla.csv')
cliente_tabla['First'] = cliente_tabla['NombreCliente'].apply(lambda x: x.split(' ')[0])
temp = dict()
tnp = cliente_tabla['First'].unique()
for i in range(tnp.shape[0]):
    temp[tnp[i]] = i
cliente_tabla['MapFirst'] = cliente_tabla['First'].apply(lambda x: temp[x])
temp = cliente_tabla[['Cliente_ID', 'MapFirst']].values
cliente_cluster = dict()
for i in range(temp.shape[0]):
    if cliente_cluster.get(temp[i,0], -1) == -1:
        cliente_cluster[temp[i,0]] = temp[i,1]
    else:
        pass
PC = producto_cluster           # producto dict
AC = town_state_cluster         # agencia dict
CC = cliente_cluster            # cliente dict

# Normalize train
def chufa(x, y):
    ret = 0
    if abs(y) > 1e-6:
        ret = x / y
    return ret

dtypes_train = {#'Semana': np.int8, 
                'Agencia_ID': np.int16, 'Canal_ID': np.int8, 
                'Ruta_SAK' : np.int16, 'Cliente_ID':np.uint32, 
                'Producto_ID': np.uint16, 'Demanda_uni_equil': np.int16}
               
dtypes_test = {#'Semana': np.int8, 
               'Agencia_ID': np.int16, 'Canal_ID': np.int8, 
               'Ruta_SAK' : np.int16, 'Cliente_ID':np.uint32, 
               'Producto_ID': np.uint16}


reader = pd.read_csv('../input/train.csv', dtype = dtypes_train, 
                     usecols = dtypes_train.keys(), chunksize = 100000)
cnt = 0
df_train = pd.DataFrame()
for chunk in reader:
    chunk['Agencia_ID'] = chunk['Agencia_ID'].apply(lambda x: AC[x])
    chunk['Producto_ID'] = chunk['Producto_ID'].apply(lambda x: PC[x])
    chunk['Cliente_ID'] = chunk['Cliente_ID'].apply(lambda x: CC[x])
    print(type(chunk))
    df_train = df_train.append(chunk, ignore_index=True)
    cnt += 1
    if cnt >= 2:
        break
    
# df_train = pd.read_csv('../input/train.csv', dtype = dtypes_train, usecols = dtypes_train.keys())
# df_train['Agencia_ID'] = df_train['Agencia_ID'].apply(lambda x: AC[x])
# df_train['Producto_ID'] = df_train['Producto_ID'].apply(lambda x: PC[x])
# df_train['Cliente_ID'] = df_train['Cliente_ID'].apply(lambda x: CC[x])
print(df_train.head())
print(df_train.shape)

# df_test = pd.read_csv('../input/test.csv', dtype = dtypes_test)
# df_test['Agencia_ID'] = df_test['Agencia_ID'].apply(lambda x: AC[x])
# df_test['Producto_ID'] = df_test['Producto_ID'].apply(lambda x: PC[x])
# df_test['Cliente_ID'] = df_test['Cliente_ID'].apply(lambda x: CC[x])
# print(df_test.head())

groupkeys = [#'Semana', 
             'Agencia_ID', 'Canal_ID','Ruta_SAK', 'Cliente_ID', 'Producto_ID']

# train = df_train.groupby(groupkeys)['Demanda_uni_equil'].mean()
# print(train.head())