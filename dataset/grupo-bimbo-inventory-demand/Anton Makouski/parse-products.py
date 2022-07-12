import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv",
                            index_col = 0,
                            nrows=   50000000, #50000000, 
                            dtype  = {'Semana': np.uint8,
                                     'Agencia_ID' : np.uint16,
                                     'Canal_ID' : np.uint8,
                                     'Ruta_SAK' : np.uint16,
                                     'Cliente_ID' : np.uint32,
                                     'Producto_ID': np.uint16,
                                     'Venta_uni_hoy': np.uint16,
                                     'Venta_hoy': np.float32,
                                     'Dev_uni_proxima': np.uint16,
                                     'Dev_proxima': np.float32,
                                     'Demanda_uni_equil': np.uint32})
#test = pd.read_csv("../input/test.csv")

products  =  pd.read_csv("../input/producto_tabla.csv", index_col = 0)

products['name'] = products.NombreProducto
# remove ' '  between digits
sn =  products.NombreProducto.str.extract('(.*[ |\D])(\d+) (\d+)(.*)', expand=True)
products.loc[pd.notnull(sn[0]), 'name'] = (sn[0]+sn[1]+sn[2]+sn[3])[pd.notnull(sn[0])]
#replace ' '  between digits by '.' for Kg
sn =  products.NombreProducto.str.extract('(.*[ |\D])(\d+) (\d+)(Kg|g8oz)(.*)', expand=True)
products.loc[pd.notnull(sn[0]), 'name'] = (sn[0]+sn[1]+'.'+sn[2]+sn[3]+sn[4])[pd.notnull(sn[0])]

spq =  products.name.str.extract('(.*) (\d+)(pq|Pq)', expand=True)
products['package'] = spq[1].astype('float')
products['short_name'] = spq[0]

sp =  products.name.str.extract('(.*) (\d+)(p|P)( |\d)', expand=True)
products['pieces'] = sp[1].astype('float')
idx = pd.isnull(products['short_name']) | (products['short_name']>sp[0])
products.loc[idx, 'short_name'] = sp[0][idx]

sw = products.name.str.extract('(.*?[\D+?| ])([\d|\.]+)(Kg|kg|g|G|ml| ml)', expand=False)
products['weight'] = sw[1].astype('float')*sw[2].map({'Kg':1000, 'kg':1000, 'G':1, 'g':1, 'ml':1, ' ml':1})
idx = pd.isnull(products['short_name']) | (products['short_name']>sw[0])
products.loc[idx, 'short_name'] = sw[0][idx]

sb = products.name.str.extract('(.*?) ?([A-Z]*) ([A-Z]+) \d+$', expand=False)
products['sub_brand'] = sb[1]
products['brand'] = sb[2]
idx = pd.isnull(products['short_name']) | (products['short_name']>sb[0])
products.loc[idx, 'short_name'] = sb[0][idx]

prices = train.pivot_table(['Venta_uni_hoy','Venta_hoy','Dev_uni_proxima','Dev_proxima'], index='Producto_ID', aggfunc=np.sum)
prices['price'] = (prices['Venta_hoy']+prices['Dev_proxima']) / (prices['Venta_uni_hoy']+prices['Dev_uni_proxima'])

products = pd.concat([products,prices['price']], axis=1)