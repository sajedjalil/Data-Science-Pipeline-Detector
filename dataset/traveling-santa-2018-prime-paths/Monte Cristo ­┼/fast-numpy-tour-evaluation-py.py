# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from sympy import isprime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

def eval_tour(tour:list, xy_citites_as_numpy_array):
    primes_penalty = np.array(list(map(isprime, tour)), 'int')*0.1

    position_penalty = np.arange((len(tour)), dtype='int')+1
    position_penalty = np.where(position_penalty%10 == 0, 1.1, 1)

    penalty = (position_penalty-primes_penalty).clip(1, 1.1)
    return (np.sqrt((np.diff(xy_citites_as_numpy_array[tour+[0]], axis=0)**2).sum(axis=1))*penalty).sum()
    
    