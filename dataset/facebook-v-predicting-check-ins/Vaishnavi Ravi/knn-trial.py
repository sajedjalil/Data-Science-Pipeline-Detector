import numpy as np
import pandas as pd

df_train = pd.read_csv('../input/train.csv',
                       usecols=['row_id','x','y','time','place_id'], 
                       index_col = 0)
                       
df_train.head(5)