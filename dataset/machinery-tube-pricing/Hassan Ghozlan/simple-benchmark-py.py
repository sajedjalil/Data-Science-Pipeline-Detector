# Simple benchmark, a bit better than all zeros
# Make the same guess for all entries.
# Choose the best constant price (in the sense of minimizing the RMSLE)

import pandas as pd
import numpy as np
from numpy import log, exp
path="../input/" # path to data
train = pd.read_csv(path+"train_set.csv",header=0,index_col=None)
p_tilde_star = np.mean(log(train['cost']+1))
#relation between p_tilde and p is: p_tilde = log(p+1)
p_star = exp(p_tilde_star)-1 #8.02932430085
test = pd.read_csv(path+"test_set.csv",header=0,index_col=None)
test['cost'] = p_star
submit = test[['id','cost']]
submit.to_csv("mysubmit-const.csv",index=False)

import os
os.system("echo Best constant guess for price is "+str(p_star))