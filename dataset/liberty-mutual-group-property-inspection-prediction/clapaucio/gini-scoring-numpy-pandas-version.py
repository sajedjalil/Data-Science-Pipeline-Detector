# Simple implementation of the (normalized) gini score in numpy
# Fully vectorized, no python loops, zips, etc.
# Significantly (>30x) faster than previous implementions

import numpy as np 
import pandas as pd

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest) with a second key to sort ambiguities
    df = pd.DataFrame({'y_true':y_true,'y_pred':y_pred, 'count':np.arange(n_samples)})
        
    true_order = df.sort(columns=['y_true','count'], ascending=(False,True))['y_true']
    pred_order = df.sort(columns=['y_pred','count'], ascending=(False,True))['y_true']


    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true