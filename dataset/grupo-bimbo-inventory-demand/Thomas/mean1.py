import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#timing = pd.read_csv('../input/train.csv', usecols=['Semana','Demanda_uni_equil'])

#timing = timing.sample(1000000)
#timing = timing.loc[timing['Demanda_uni_equil'] < 15]

#x = timing['Semana'].tolist()
#y = timing['Demanda_uni_equil'].tolist()

#plt.hist2d(x, y, bins=[7, 15])
#plt.label_plot('Distribution of target value over time', 'Week', 'Target')
#plt.savefig("hist2.png")

#items = pd.read_csv('../input/train.csv', usecols=['Producto_ID'])
#target = items.tolist()

#plt.hist(target, bins=200, color='blue')
#label_plot('Distribution of target values', 'Demanda_uni_equil', 'Count')
#plt.show()
#plt.savefig("items.png")


df_train = pd.read_csv('../input/train.csv', usecols=['Demanda_uni_equil'], nrows=100)
print(np.mean(df_train))
#df_train = pd.read_csv('../input/train.csv', nrows=500)

#pd.DataFrame.hist(df_train, column="Producto_ID", bins=100)

##df_train.plot(x='Semana',y='Demanda_uni_equil')
#plt.savefig("items.png")

#from collections import Counter
#print(Counter(df_train['Producto_ID']).most_common(10))

#idxs = df_train['Producto_ID'] == 1250
#pd.DataFrame.hist(df_train[idxs], column="Demanda_uni_equil", bins=100)
#print(df_train[idxs]['Demanda_uni_equil'])
#print(df_train.iloc(479414))
#plt.savefig("items.png")

#print(items[:10])
#print(items == 1212)
#print(Counter(items).most_common(10))
#print('Our most common value is 2')
