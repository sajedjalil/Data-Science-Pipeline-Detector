'''
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
print(train.columns)

train_data = train.values

agg  = train.groupby(['Semana', 'Producto_ID'], as_index=False).agg(['count','sum', 'min', 'max','median','mean'])
agg.columns  =  ['_'.join(col).strip() for col in agg.columns.values]
print(agg)

#print(train[['Producto_ID','Semana']])
#print(train[(train['Producto_ID']==1212)][['Producto_ID','Semana','Demanda_uni_equil']])
'''

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
output_notebook()

def get_product_agg(cols):
    train = pd.read_csv('../input/train.csv', usecols = ['Semana', 'Producto_ID'] + cols,
                        dtype  = {'Semana': 'int32',
                                 'Producto_ID':'int32',
                                 'Venta_hoy':'float32',
                                 'Venta_uni_hoy': 'int32',
                                 'Dev_uni_proxima':'int32',
                                 'Dev_proxima':'float32',
                                 'Demanda_uni_equil':'int32'})
    agg  = train.groupby(['Semana', 'Producto_ID'], as_index=False).agg(['count','sum', 'min', 'max','median','mean'])
    agg.columns  =  ['_'.join(col).strip() for col in agg.columns.values]
    del(train)
    return agg
    
agg1 = get_product_agg(['Demanda_uni_equil','Dev_uni_proxima'])
print(agg1)