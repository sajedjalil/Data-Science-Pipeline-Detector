import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open('../input/s1-dropnancolumnslist/col_list.pickle','rb') as f:
    col_list = pickle.load(f)

with open('../input/s1-dropnancolumnslist/dtyps.pickle','rb') as f:
    dtypes = pickle.load(f)

print('reading data...')   
df = pd.read_csv('../input/microsoft-malware-prediction/train.csv', usecols=col_list, dtype=dtypes)

print('pickling data...')
with open('df_nancols_dropped.pickle','wb') as f:
    pickle.dump(df, f)
    
print('done.')
    
    
    
    
