# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print ('Reading train!')
train = pd.read_csv('../input/train.csv',
                    usecols=['Agencia_ID',
                                  'Ruta_SAK',
                                  'Cliente_ID',
                                  'Producto_ID',
                                  'Demanda_uni_equil'],
                    dtype={'Agencia_ID': 'uint16',
                                      'Ruta_SAK': 'uint16',
                                      'Cliente_ID': 'int32',
                                      'Producto_ID': 'uint16',
                                      'Demanda_uni_equil': 'float32'})
print ('Train read!')
print ('Estimating means features')
train['Demanda_uni_equil'] = train['Demanda_uni_equil'].apply(lambda x: np.log1p(x))
mean = train['Demanda_uni_equil'].mean()
print ('Transformed DuE')
train['MeanP'] = train.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
print ('Got MeanP')
train['MeanC'] = train.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
print ('Got MeanC')
train['MeanPA'] = train.groupby(['Producto_ID',
                                 'Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
print ('Got MeanPA')
train['MeanPR'] = train.groupby(['Producto_ID',
                                 'Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
print ('Got MeanPR')
train['MeanPCA'] = train.groupby(['Producto_ID',
                                  'Cliente_ID',
                                  'Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
print ('Got MeanPCA')
train.drop_duplicates(subset=['Agencia_ID', 'Cliente_ID', 'Ruta_SAK', 'Producto_ID'], inplace=True)

print ('Dropping duplicates!')
MPCA = train.loc[:, ['Producto_ID', 'Agencia_ID', 'Cliente_ID', 'MeanPCA']].drop_duplicates(subset=['Agencia_ID', 'Cliente_ID', 'Producto_ID'])
print ('1')
MPA = train.loc[:, ['Producto_ID', 'Agencia_ID', 'MeanPA']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID'])
print ('2')
MC = train.loc[:, ['Cliente_ID','MeanC']].drop_duplicates(subset=['Cliente_ID'])
print ('3')
MP = train.loc[:, ['Producto_ID', 'MeanP']].drop_duplicates(subset=['Producto_ID'])
print ('4')
MPR = train.loc[:, ['Producto_ID', 'Ruta_SAK', 'MeanPR']].drop_duplicates(subset=['Ruta_SAK', 'Producto_ID'])
print ('5')
gc.collect()
print ('Reading Test')
test = pd.read_csv('../input/test.csv',
                   usecols=['Agencia_ID',
                              'Ruta_SAK',
                              'Cliente_ID',
                              'Producto_ID',
                            'id'],
                   dtype={'Agencia_ID': 'uint16',
                                      'Ruta_SAK': 'uint16',
                                      'Cliente_ID': 'int32',
                                      'Producto_ID': 'uint16',
                                      'id': 'int32'})
print ('Test read!')
print ('Merging!')
test = test.merge(MPCA,
                  how='left',
                  on=['Producto_ID', 'Agencia_ID', 'Cliente_ID'],
                  copy=False)
print ('PCA merged!')
test = test.merge(MPA,
                  how='left',
                  on=['Producto_ID', 'Agencia_ID'],
                  copy=False)
print ('PA merged!')
test = test.merge(MC,
                  how='left',
                  on=['Cliente_ID'],
                  copy=False)
print ('C merged!')
test = test.merge(MP,
                  how='left',
                  on=['Producto_ID'],
                  copy=False)
print ('P merged')
test = test.merge(MPR,
                  how='left',
                  on=['Producto_ID', 'Ruta_SAK'],
                  copy=False)
print ('PR merged')

print ('Merging done!')
gc.collect()
print (test.shape)
print (test.columns)