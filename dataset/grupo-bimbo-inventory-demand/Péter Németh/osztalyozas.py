#2. hét hétfő 2. feladat 
import pandas as pd
import numpy as np
import re, time
import random
start_cpu = time.clock()
start_real = time.time()

df = pd.read_csv('../input/train.csv',nrows=300000)


def vtln(row):
    return random.randint(0,9)
df['osztaly']=df.apply(lambda row:vtln(row),axis=1)
def evaluate(df):
	eval=0
	rowc=df.shape[0]
	for row in df.itertuples():
		a=np.log1p(row.tipp)-np.log1p(row.Demanda_uni_equil)
		eval+=a**2
	eval *= 1/rowc
	return eval
ertekek=[]
for i in range(0,10):
    df_tmp=df[df.osztaly==i].copy()
    demand_mean=df_tmp.Demanda_uni_equil.mean()
    df_tmp.loc[:,'tipp'] = pd.Series(demand_mean, index=df_tmp.index)
    ertekek.append(evaluate(df_tmp))
print(ertekek)

elapsed_cpu = time.clock() - start_cpu
elapsed_real = time.time() - start_real
print('CPU idő:%s, Valós idő:%s'%(elapsed_cpu,elapsed_real) )
'''
    df_train = pd.read_csv('../input/train.csv,usecols=['Producto_ID','Cliente_ID','Demanda_uni_equil'],nrows=300)
    train_global_median=df_train.Demanda_uni_equil.median()
    df_tmp.loc[:,'tipp'] = pd.Series(train_global_median, index=df.index)
    df_cliente_median = df_train.loc[:,['Cliente_ID','Demanda_uni_equil']].groupby(['Cliente_ID'],as_index=False).median().rename(index=str, columns={"Demanda_uni_equil": "Dem1"})
    df_producto_median = df_train.loc[:,['Producto_ID','Demanda_uni_equil']].groupby(['Producto_ID'],as_index=False).median().rename(index=str, columns={"Demanda_uni_equil": "Dem2"})
    df_pr_cl_median =  df_train.groupby(['Producto_ID','Cliente_ID'],as_index=False).median().rename(index=str, columns={"Demanda_uni_equil": "Dem3"})
    df_tmp = pd.merge(df_tmp, df_cliente_median, how='left', on=['Cliente_ID'])
    df_tmp = pd.merge(df_tmp, df_producto_median, how='left', on=['Producto_ID'])
    df_tmp = pd.merge(df_tmp, df_pr_cl_median, how='left', on=['Producto_ID', 'Cliente_ID'])
    df_tmp = pd.merge(df_tmp, df_train, how='left')
    def egyesit(row):
        # Producto-Cliente medián
        if(not np.isnan(row['Dem3'])):
            return row['Dem3']
        # Producto medián
        if(not np.isnan(row['Dem2'])):
            return row['Dem2']
        # Cliente medián
        if(not np.isnan(row['Dem1'])):
            return row['Dem1']
        # Globális medián
        return row['tipp']
    df_tmp['tipp'] = df_tmp.apply(lambda row:egyesit(row) ,axis=1)
    del df_tmp['Dem1']
    del df_tmp['Dem2']
    del df_tmp['Dem3']
    print(df_tmp)
'''