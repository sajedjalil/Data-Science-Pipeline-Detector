#..written for python 2.7
import numpy as np 
import pandas as pd

def CRPS_row(row):
    """
    This function is purpose-built for Kaggle. With a couple tweaks it can 
    be generalized for other applications. 
    
    row should be a 601-element list where the first element is the true volme 
    and the subsequent elements are cumulatively summed probabilities. 
    """
    V_m = row[0]
    p = np.array(row[1:])
    v = np.array(range(len(n)))
    h = v >= V_m
    sq_dists = (p - h)**2
    return(np.sum(sq_dists)/len(sq_dists)) 

def CRPS_mean(df): 
    """
    Function recieves pandas dataframe as an input with the first column being
    a column of truths (V_m) and the subsequent 600 columns being cumulatively
    summed probabilities
    """
    crps_vec = df.apply(CRPS_row, axis = 1)
    crps_sum = np.sum(crps_vec)
    return(crps_sum/len(crps_vec))

