import time
start_cpu = time.clock()
start_real = time.time()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
def elemez(be):
    df_tmp = pd.read_csv(be, nrows=1)
    n= (df_tmp.shape[1])
    
    i=0
    print('--"'+be+'"--')
    while (i<n):
        df = pd.read_csv(be,usecols=[i])
        print(str(i)+'. sor('+df_tmp.iloc[:,i].name+') :')
        print('Null-értékek')
        print(df.isnull().sum())
        print('Rekordok száma')
        print(df.count())
        print('Numerikus oszlop minimuma, maximuma, mediánja és átlaga')
        print(df.min())
        print(df.max())
        print(df.median())
        print(df.mean())
        i+=1
        del(df)
    del(df_tmp)
#elemez('../input/test.csv')
#elemez('../input/train.csv')
def becsul(be_train,be_teszt):
    df_Train = pd.read_csv(be_train,usecols=['Producto_ID','Cliente_ID','Demanda_uni_equil'])
    df = pd.read_csv(be_teszt,usecols=['Producto_ID','Cliente_ID'])
    train_global_median=df_Train.Demanda_uni_equil.median()
    #print(train_global_median)
    df.loc[:,'Demanda_uni_equil'] = pd.Series(train_global_median, index=df.index)
    df_cliente_median = df_Train.loc[:,['Cliente_ID','Demanda_uni_equil']].groupby(['Cliente_ID'],as_index=False).median().rename(index=str, columns={"Demanda_uni_equil": "Dem1"})
    df_producto_median = df_Train.loc[:,['Producto_ID','Demanda_uni_equil']].groupby(['Producto_ID'],as_index=False).median().rename(index=str, columns={"Demanda_uni_equil": "Dem2"})
    df_pr_cl_median =  df_Train.groupby(['Producto_ID','Cliente_ID'],as_index=False).median().rename(index=str, columns={"Demanda_uni_equil": "Dem3"})
    #print(df_pr_cl_median)
    #print(df.head(10))
    #print(df.dtypes)
    #print(df_pr_cl_median.dtypes)
    #df_cliente_median.merge(df_pr_cl_median,on=['Cliente_ID'])
    
    #print(df.head(20))
    df = pd.merge(df, df_cliente_median, how='left', on=['Cliente_ID'])
    df = pd.merge(df, df_producto_median, how='left', on=['Producto_ID'])
    df = pd.merge(df, df_pr_cl_median, how='left', on=['Producto_ID', 'Cliente_ID'])
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
        return row['Demanda_uni_equil']
    df['Demanda_uni_equil'] = df.apply(lambda row:egyesit(row) ,axis=1)
    del df['Dem1']
    del df['Dem2']
    del df['Dem3']
    print(df)
    print(df.columns)
    df['id'] = df.index
    df_submit = df[['id', 'Demanda_uni_equil']]
    df_submit = df_submit.set_index('id')
    df_submit.to_csv('naive_product_client_median.csv')
becsul('../input/train.csv','../input/test.csv')
#['Producto_ID','Agencia_ID']
def csutortok_elso_masodik(be_train,be_teszt):
    df_pr_train = pd.read_csv(be_train,usecols=['Producto_ID']).drop_duplicates()
    df_pr = pd.read_csv(be_teszt,usecols=['Producto_ID']).drop_duplicates()
    df_pr_train.loc[:,'megvan'] = pd.Series(1, index=df_pr_train.index)
    df_pr = pd.merge(df_pr, df_pr_train, how='left', on=['Producto_ID'])
    df_pr=df_pr[df_pr.megvan.isnull()]
    del df_pr['megvan']
    print("Product_ID, ami a test-ben benne van, a train-ben viszont nincs: "+str(df_pr.Producto_ID.count()))
    df_ag_train = pd.read_csv(be_train,usecols=['Agencia_ID']).drop_duplicates()
    df_ag = pd.read_csv(be_teszt,usecols=['Agencia_ID']).drop_duplicates()
    df_ag_train.loc[:,'megvan'] = pd.Series(1, index=df_ag_train.index)
    df_ag = pd.merge(df_ag, df_ag_train, how='left', on=['Agencia_ID'])
    df_ag=df_ag[df_ag.megvan.isnull()]
    del df_ag['megvan']
    print("Sales_Depot_ID, ami a test-ben benne van, a train-ben viszont nincs: "+str(df_ag.Agencia_ID.count()))
    print("Sales_Depot_ID-k a train halmazban: "+str(df_ag_train.Agencia_ID.count()))
#csutortok_elso_masodik('../input/train.csv','../input/test.csv')    
#Csütörtök harmadik pont, vázlat

def csutortok_harmadik_negyedik(be_train,oszlop):
    df_train = pd.read_csv(be_train,usecols=[oszlop,'Demanda_uni_equil'])
    df_max = df_train.loc[:,[oszlop,'Demanda_uni_equil']].groupby([oszlop],as_index=False).max().rename(index=str, columns={"Demanda_uni_equil": "maximum"})
    df = df_train.loc[:,[oszlop,'Demanda_uni_equil']].groupby([oszlop],as_index=False).min().rename(index=str, columns={"Demanda_uni_equil": "minimum"})
    df = pd.merge(df, df_max, how='left', on=[oszlop])
    df['elteres'] = df.apply(lambda row:row['maximum']-row['minimum'] ,axis=1)
    df=df.sort_values(by='elteres',ascending=False)
    print('Az elso 20 legnagyobb elteres '+oszlop+' oszlopoknal')
    print(df.head(20))
    df_elt= df.loc[:,['elteres',oszlop] ].groupby(['elteres'],as_index=False).count().rename(index=str, columns={oszlop: "hany_darab"})
    print('Adott szamu eltereshez hany '+oszlop+' tartozik:')
    print(df_elt)
#csutortok_harmadik_negyedik('../input/train.csv','Cliente_ID')
#csutortok_harmadik_negyedik('../input/train.csv','Agencia_ID')
def pentek_klaszter(be_train):
    from sklearn.cluster import KMeans
    df_train = pd.read_csv(be_train,nrows=300)
    y_pred = KMeans(n_clusters=10).fit_predict(df_train)
    df_train['klaszter']=y_pred
    print(df_train)
#pentek_klaszter('../input/train.csv')
elapsed_cpu = time.clock() - start_cpu
elapsed_real = time.time() - start_real
print('CPU idő:%s, Valós idő:%s'%(elapsed_cpu,elapsed_real) )