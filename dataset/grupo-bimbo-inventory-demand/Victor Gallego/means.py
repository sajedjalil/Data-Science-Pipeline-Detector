# Means. Group Bimbo Inventory Demand competition in Kaggle.

import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def load_train_data():
    dir_train = '../input/train.csv'
    df_train = pd.read_csv(dir_train, usecols=[ 'Semana', 'Cliente_ID', 'Producto_ID', 'Ruta_SAK', 'Demanda_uni_equil'], 
                           dtype  = {'Semana': 'uint8',
                                     'Producto_ID':'uint32',
                                     'Cliente_ID' : 'uint32',
                                     'Demanda_uni_equil':'uint32'})
    return df_train

def load_test_data():
    dir_test = '../input/test.csv'
    df_test = pd.read_csv(dir_test, usecols=[ 'id', 'Cliente_ID', 'Producto_ID', 'Ruta_SAK' ],
                          dtype  = { 'Semana': 'uint8',
                                     'Producto_ID':'uint32',
                                     'Cliente_ID' : 'uint32',
                                     'id':'uint32'})
    return df_test


df = load_train_data()

n_semanas = 7 # To choose a subset of the training dataset
df = df[df.Semana >= 3 + (7 - n_semanas)]

# Since the score is given by RMSLE, we transform the values using log(1+x)
df['Demanda_uni_equil'] = df['Demanda_uni_equil'].apply(np.log1p)

# Compute the means of the target variable 'Demanda_uni_equil', grouping
# by some subsets of variables
mean = df.Demanda_uni_equil.mean()
meanP = df.groupby(['Producto_ID'], as_index=False)['Demanda_uni_equil'].mean()
#meanPC = df.groupby(['Cliente_ID', 'Producto_ID'], as_index=False)['Demanda_uni_equil'].mean()
meanPR = df.groupby(['Producto_ID', 'Ruta_SAK'], as_index=False)['Demanda_uni_equil'].mean()
#meanPCR = df.groupby(['Cliente_ID', 'Producto_ID', 'Ruta_SAK'], as_index=False).mean()

test = load_test_data()
#test = test.merge(meanPCR, on=['Cliente_ID', 'Producto_ID', 'Ruta_SAK'], how='left')
#test = test.merge(meanPC, on=['Cliente_ID', 'Producto_ID'], how='left')
test = test.merge(meanPR, on=['Ruta_SAK', 'Producto_ID'], how='left')
test = test.merge(meanP, on=['Producto_ID'], how='left')

# If some value is NaN, we use the next mean to predict the value
test['Demanda_uni_equil'] = 0.74198*test.Demanda_uni_equil_x-0.00334558
#test['Demanda_uni_equil'] = test['Demanda_uni_equil'].fillna(0.708135*test['Demanda_uni_equil_y']+0.000529789)
test['Demanda_uni_equil'] = test['Demanda_uni_equil'].fillna(test['Demanda_uni_equil_y'])
#test['Demanda_uni_equil'] = test['Demanda_uni_equil'].fillna(test['Demanda_uni_equil_z'])
test['Demanda_uni_equil'] = test['Demanda_uni_equil'].fillna(mean)
#test['Demanda_uni_equil'] = test['Demanda_uni_equil'].fillna(0.681167*mean+0.000596246)

test['Demanda_uni_equil'] = test['Demanda_uni_equil'].apply(np.expm1)


# Output
test.to_csv('submission_means.csv', columns=['id', 'Demanda_uni_equil'], index=False)

# Any results you write to the current directory are saved as output.