import pandas as pd
import numpy as np

types = {'Semana': np.uint8,'Cliente_ID': np.uint16, 'Producto_ID': np.uint16,
'Demanda_uni_equil': np.uint8}
types2 = {'Cliente_ID': np.uint16, 'Producto_ID': np.uint16}

dftrain = pd.read_csv('../input/train.csv', usecols=types.keys(), dtype=types)
dftest = pd.read_csv('../input/test.csv',skiprows=0, nrows=1000)
print(dftest.head(2))
dftest['predict']=""
numrows =0
print(dftest.head(2))
for index, row in dftest.iterrows():
    if(numrows%100==0):
        print(numrows)
    numrows+=1
    week = int(row[1])
    ClientId = int(row[5])
    ProductID = int(row[6])
    avg1=0
    avg2=0
    avg3=0
    count1=0
    count2=0
    count3=0
    if week==10:
        
        temp = dftrain[dftrain['Semana'] == week-1]
        temp2 = temp[temp['Cliente_ID'] == ClientId]
        temp3=temp2[temp2['Producto_ID']== ProductID]
        if not temp3.empty:
            avg1 = temp3["Demanda_uni_equil"].mean()
            count1=len(temp3["Demanda_uni_equil"])
    
        dftest.loc[index, 'predict'] = avg1+avg2+avg3
        
    if week==11:
    
        temp = dftrain[dftrain['Semana'] == week-2]
        temp2 = temp[temp['Cliente_ID'] == ClientId]
        temp3=temp2[temp2['Producto_ID']== ProductID]
        if not temp3.empty:
            avg1 = temp3["Demanda_uni_equil"].mean()
            count1=len(temp3["Demanda_uni_equil"])
    
        dftest.loc[index, 'predict'] = avg1+avg2+avg3
        
dftest.to_csv("2.csv",index = False)    
