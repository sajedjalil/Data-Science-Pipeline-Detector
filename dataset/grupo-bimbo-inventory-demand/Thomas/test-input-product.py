import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df_train = pd.read_csv('../input/train.csv', nrows=500000)
#df_train = pd.read_csv('../input/train.csv')

print(df_train[:10])

#file_in = '../input/train.csv'
#tp = pd.read_csv(file_in, iterator=True, chunksize=1000, usecols=['Demanda_uni_equil'])
#df_train = pd.concat(tp, ignore_index=True)
#target = df_train['Demanda_uni_equil'].tolist()

timing = pd.read_csv('../input/train.csv', usecols=['Producto_ID','Demanda_uni_equil'])
#print(timing[:10])
target = timing['Producto_ID'].tolist()


#plt.hist(target, bins=200, color='blue', range=(0, 50))
plt.hist(target, bins=200, color='blue', range=(0, 500))
#label_plot('Distribution of target values', 'Demanda_uni_equil', 'Count')
#plt.show()
plt.savefig("test_product.png")

#sub = pd.read_csv('../input/sample_submission.csv')
#sub['Demanda_uni_equil'] = 2.0
#sub.to_csv('mostcommon.csv', index=False)