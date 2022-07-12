import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df_train = pd.read_csv('../input/train.csv', usecols=['Producto_ID','Demanda_uni_equil'])
df_test = pd.read_csv('../input/test.csv', usecols=['Producto_ID'])

sub = pd.read_csv('../input/sample_submission.csv')
#set most common value
sub['Demanda_uni_equil'] = 2.0 

#find the most common values for the top N products in the training set
N = 10
top_N = Counter(df_train['Producto_ID']).most_common(N)
results = {}
for item, value in top_N:
    idx = df_train['Producto_ID'] == item
    most_common_value = Counter(df_train.loc[idx, 'Demanda_uni_equil']).most_common(1)[0][0]
    results[item] = most_common_value
    
    #find the locations of these prodcuts in the test set
    idx = df_test['Producto_ID'] == item
    #assign these prodcuts their most common values
    sub.loc[idx, 'Demanda_uni_equil'] = most_common_value 
    
sub.to_csv('mostcommon.csv', index=False)