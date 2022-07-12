# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

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