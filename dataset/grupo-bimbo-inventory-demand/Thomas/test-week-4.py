import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd

#data = pd.read_csv('../input/train.csv', usecols=['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID'])
#data = pd.read_csv('../input/train.csv', usecols=['Agencia_ID', 'Canal_ID'], nrows=500000)
data = pd.read_csv('../input/train.csv', usecols=['Agencia_ID', 'Producto_ID'], nrows=500000)

#x = data['Agencia_ID'].tolist()
#y = data['Canal_ID'].tolist()

from collections import Counter
d = [a for a,b in Counter(data['Agencia_ID']).most_common(10)]

for agent in d:
    idx = data['Agencia_ID'] == agent
    want = data.loc[idx,'Producto_ID']
    plt.hist(want, bins=10, color='blue')
    plt.savefig("%d.png"%agent)
    plt.clf()


#plt.hist(x, bins=100, color='blue')
#plt.savefig("agencia.png")

#plt.hist(y, bins=10, color='blue')
#plt.savefig("canal.png")

#lt.hist2d(x, y, bins=[50, 10])
#plt.savefig("agencia-canal.png")

#plt.hist(data['Cliente_ID'].tolist(), bins=100, color='blue')
#plt.savefig("cliente.png")


#idx = data['Semana'] == 4
#want = data.loc[idx, ['Venta_uni_hoy','Dev_uni_proxima','Demanda_uni_equil']]

#print(want[:60])
#want['ans'] = want['Venta_uni_hoy'] - want['Venta_uni_hoy']

#plt.plot(want['ans'].tolist())
#plt.savefig("4.png")


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


#df_train = pd.read_csv('../input/train.csv', nrows=500000)
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
