import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import OneHotEncoder

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
town = pd.read_csv('../input/town_state.csv')
town.info()
tn = pd.concat([town.Agencia_ID,
                pd.get_dummies(town.State),
                pd.get_dummies(town.Town)],
              axis=1)
tn.to_csv('town_encoded.csv', index=False)
products = pd.read_csv('../input/producto_tabla.csv')
products.info()
products['grams'] = products.NombreProducto.str.extract('.* (\d+)g.*', expand=False)
products['ml'] = products.NombreProducto.str.extract('.* (\d+)ml.*', expand=False)
products['inches'] = products.NombreProducto.str.extract('.* (\d+)in.*', expand=False)
products['pct'] = products.NombreProducto.str.extract('.* (\d+)pct.*', expand=False)
products['pieces'] = products.NombreProducto.str.extract('.* (\d+)p.*', expand=False)
labels = products.NombreProducto.str.extract('([^\d]+) \d+.*', expand=False)
pr = pd.concat([products.drop('NombreProducto', axis=1),
                pd.get_dummies(labels)],
               axis=1)
pr.info()
pr.to_csv('products_encoded.csv', index=False)
clients = pd.read_csv('../input/cliente_tabla.csv')
clients.info()
clients.NombreCliente.replace(['SIN NOMBRE',
                               'NO IDENTIFICADO'], np.nan, inplace=True)
clients.info()
clients.to_csv('clients_encoded.csv', index=False)
