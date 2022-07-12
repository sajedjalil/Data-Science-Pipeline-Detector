import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt

types = {'Semana':np.uint8, 'Agencia_ID':np.uint16, 'Canal_ID':np.uint8,
         'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 'Producto_ID':np.uint16,
         'Demanda_uni_equil':np.uint32}

train = pd.read_csv('../input/train.csv', usecols=types.keys(), dtype=types)

g = train.groupby(['Cliente_ID', 'Producto_ID'])
gcounts = g.count()
print(gcounts.Semana.describe())

#sns.distplot(gcounts.Semana)
#plt.show()

print('max counts for client and product ID:', gcounts.Semana.max())

#print(gcounts.ix[gcounts.Semana.idxmax()])

#cli40853 = train[train['Cliente_ID']==653378]
#print(cli40853[cli40853['Producto_ID']==42128])

uniqueWeeks = g.Semana.nunique()
sns.distplot(uniqueWeeks, kde=False)
plt.savefig('uniqueWeeks.png')