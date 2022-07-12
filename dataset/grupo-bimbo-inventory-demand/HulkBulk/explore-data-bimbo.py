import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



df_train = pd.read_csv('../input/train.csv', nrows=100000)

#print(df_train)
#print(df_test)
df_cliente =pd.read_csv('../input/cliente_tabla.csv')
#print(df_cliente)
df_producto = pd.read_csv('../input/producto_tabla.csv')
df_town = pd.read_csv('../input/town_state.csv')


def merging(df):
    df = pd.merge(df, df_cliente, on='Cliente_ID')
    df = pd.merge(df, df_producto, on='Producto_ID')
    df = pd.merge(df, df_town, on='Agencia_ID')
    return df

#df_train = merging(df_train)
label_reg = df_train['Demanda_uni_equil']
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

df_train.drop(['Demanda_uni_equil','Dev_proxima','Dev_uni_proxima','Venta_uni_hoy','Venta_hoy'], axis=1, inplace=True)

print(df_train.dtypes)
rf = rf.fit(df_train,label_reg)


df_test = pd.read_csv('../input/test.csv')
id = df_test['id']
df_test.drop('id', axis=1, inplace=True)
print(df_test.dtypes)

s = {'id':id,'Demanda_uni_equil':rf.predict(df_test)}
pd.DataFrame(s).to_csv('sub.csv',index=False)


