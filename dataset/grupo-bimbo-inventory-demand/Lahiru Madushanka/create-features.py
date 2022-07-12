import pandas as pd
import numpy as np

types = {'Semana': np.uint8,'Cliente_ID': np.uint16, 'Producto_ID': np.uint16,
'Demanda_uni_equil': np.uint8}
types2 = {'Cliente_ID': np.uint16, 'Producto_ID': np.uint16}

dftrain = pd.read_csv('../input/train.csv', usecols=types.keys(), dtype=types)
dftest = pd.read_csv('../input/test.csv',skiprows=0, nrows=18000)
dftrain = dftrain[dftrain['Semana']>8]
dftest['predict']=""
dftrain = dftrain.sort_values(['Cliente_ID', 'Producto_ID'], ascending=[True, True])

numrows =0
print(dftest.head(2))
for index, row in dftest.iterrows():
    if(numrows%100==0):
        print(numrows)
    numrows+=1
    week = int(row[1])
    ClientId = int(row[5])
    ProductID = int(row[6])
    avg=0
    dftest1 =dftrain[(dftrain['Cliente_ID']==ClientId) & (dftrain['Producto_ID']==ProductID) & (dftrain['Semana']==9)]
    avg = dftest1[['Demanda_uni_equil']].mean()[0]
    dftest.loc[index, 'predict'] = avg
        
dftest.to_csv("1.csv")    
