# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
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
#train['Demanda_uni_equil'] = train['Demanda_uni_equil'].apply(lambda x: 1.005*np.log1p(x + 0.01) - 0.005)
mean = train['Demanda_uni_equil'].mean()
print ('Transformed DuE')
train['MeanP'] = train.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
print('Median - MeanP : ' + str(train['MeanP'].median()))
print ('Got MeanP')
train['MeanC'] = train.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
print('Median - MeanC : ' + str(train['MeanC'].median()))
print ('Got MeanC')
train['MeanPA'] = train.groupby(['Producto_ID',
                                 'Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
print('Median - MeanPA : ' + str(train['MeanPA'].median()))
print ('Got MeanPA')
train['MeanPR'] = train.groupby(['Producto_ID',
                                 'Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
print('Median - MeanPR : ' + str(train['MeanPR'].median()))
print ('Got MeanPR')
train['MeanPCA'] = train.groupby(['Producto_ID',
                                  'Cliente_ID',
                                  'Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
print('Median - MeanPCA : ' + str(train['MeanPCA'].median()))
print ('Got MeanPCA')
