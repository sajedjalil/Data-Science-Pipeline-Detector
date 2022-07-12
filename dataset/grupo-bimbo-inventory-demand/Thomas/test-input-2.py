import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df_test = pd.read_csv('../input/test.csv')
idxs = df_test['Producto_ID'] == 1250

sub = pd.read_csv('../input/sample_submission.csv')
sub['Demanda_uni_equil'] = 2.0 
sub.loc[idxs, 'Demanda_uni_equil'] = 5.0 
sub.to_csv('mostcommon.csv', index=False)