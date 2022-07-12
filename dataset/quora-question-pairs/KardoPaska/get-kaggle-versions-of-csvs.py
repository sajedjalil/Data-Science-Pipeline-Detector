# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import pickle
import gzip
import time

def prnt_updt(what_u_b_updtng, the_start):
    kardo_time = '{:*^32.22}'.format(time.strftime("%a %b %d %I:%M:%S %p"))
    elpsed = '{:,G}'.format(time.time() - the_start)
    print(kardo_time + '{:^32.32}'.format(what_u_b_updtng.upper()) + '{:<32.32}'.format(elpsed))

start_time = time.time()

prnt_updt('LOADING SHIT', start_time)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


prnt_updt('SAVING SHIT 1', start_time)
f = gzip.open('train.pklz','wb')
pickle.dump(df_train,f)
f.close()

prnt_updt('SAVING SHIT 2', start_time)
f = gzip.open('test.pklz','wb')
pickle.dump(df_test,f)
f.close()


prnt_updt('WIPE UP AND FLUSH', start_time)