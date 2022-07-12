import pandas as pd
import numpy as np

types = {'Semana': np.uint8,'Cliente_ID': np.uint16, 'Producto_ID': np.uint16,
'Demanda_uni_equil': np.uint8}
types2 = {'Cliente_ID': np.uint16, 'Producto_ID': np.uint16}

dftrain = pd.read_csv('../input/train.csv', usecols=types.keys(), dtype=types)

dftrain = dftrain[dftrain['Semana']>6].sort_values(['Cliente_ID', 'Producto_ID'], ascending=[True, True])
dftest = pd.read_csv('../input/test.csv',skiprows=0, nrows=300)
print(dftest.head(2))
dftest['predict']=""
dftest['count']=""
numrows =0
print(dftest.head(2))
for index, row in dftest.iterrows():
    if(numrows%100==0):
        print(numrows)
    numrows+=1
    week = int(row[1])
    ClientId = int(row[5])
    ProductID = int(row[6])
    predictedVal=0
   

        
    temp = dftrain[(dftrain['Cliente_ID']==ClientId) & (dftrain['Producto_ID']==ProductID) & (dftrain['Semana']==9)]
    if not temp.empty:
        predictedVal = temp["Demanda_uni_equil"].iloc[0]
        
    else:
        temp = dftrain[(dftrain['Cliente_ID']==ClientId) & (dftrain['Producto_ID']==ProductID) & (dftrain['Semana']==8)]
       
        if not temp.empty:
            predictedVal = temp["Demanda_uni_equil"].iloc[0]
        else:
            temp = dftrain[(dftrain['Cliente_ID']==ClientId) & (dftrain['Producto_ID']==ProductID) & (dftrain['Semana']==7)]
            if not temp.empty:
                predictedVal = temp["Demanda_uni_equil"].iloc[0]
            else:
                temp = dftrain[(dftrain['Producto_ID']==ProductID) & (dftrain['Semana']==9)]
                if not temp.empty:
                    predictedVal = temp["Demanda_uni_equil"].head(10).mean()
                else:
                    predictedVal = 4

    
dftest.loc[index, 'predict'] =predictedVal
        
    
        
dftest.to_csv("2.csv",index = False)