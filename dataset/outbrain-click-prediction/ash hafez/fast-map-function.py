# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

# --------------------------------------------------------------------------------
# Assumes:
#   - only one 'a' per row
#   - each of the 'p' arrays are already capped to the maximum number of predictions
#   - each label only occurs a maximum of 1 time in each prediction record 
#
def fast_map(A,P):

    return np.mean( [ 1.0 / (1 + np.where( p == a )[0][0]) for (a,p) in zip(A.ravel(), P.ravel()) ] )


# ----
# Slow Score: 0.653231 | 0.638401
# Fast Score: 0.653231 | 0.638401
#
# Slow Time:  88.765951 seconds
# Fast Time : 16.385901 seconds
# ----
