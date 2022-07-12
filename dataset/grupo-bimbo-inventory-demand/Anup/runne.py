# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

types = {'Semana': np.dtype(int),
         'Agencia_ID': np.dtype(int),
         'Canal_ID' : np.dtype(int),
         'Ruta_SAK': np.dtype(int),
         'Cliente_ID' : np.dtype(int),
         'Producto_ID' : np.dtype(int),
         'Venta_uni_hoy' : np.dtype(int),
#         'Venta_hoy' : np.dtype(float),
#         'Dev_uni_proxima' : np.dtype(int),
#         'Dev_proxima' : np.dtype(float),
         'Demanda_uni_equil' : np.dtype(int)}
          

train = pd.read_csv("../input/train.csv", usecols = ["Canal_ID", "Demanda_uni_equil"])
#train = normalize(traind, axis=0)
#print (train)

#test = pd.read_csv("../input/producto_tabla.csv", nrows = 500)
train1 = train[['Canal_ID','Demanda_uni_equil']]
#train1 = train(:,:)
#print (test.dtypes)

#print (train[0:20])
#print (train[0:20].dtypes)
#print (train.tail())
#gb1 = train.groupby(['Cliente_ID']).sum()   
#gb2 = test.groupby(['Semana','Agencia_ID'])

#for x in gb1:
#    print(x)
    
#    print(Agencia_ID)
#    print(Producto_ID)
addup1 = train1.groupby(['Canal_ID']).sum()  
#addup1n = normalize(addup1, axis=0)

print (addup1)
#var1 = addup1n.var()
#print (var1)

#train2 = train[['Cliente_ID','Demanda_uni_equil']]

#addup2 = train2.groupby(['Cliente_ID']).sum()
#print (addup2)
#var2 = addup2.var()
#print (var2)
#pca = PCA()
#pca.fit(addup)
#print(pca.explained_variance_ratio_)

#pd.DataFrame(data=train[:,:],    # values
#            index=train[:,0],    # 1st column as index
#            columns=train[0,1:]) 