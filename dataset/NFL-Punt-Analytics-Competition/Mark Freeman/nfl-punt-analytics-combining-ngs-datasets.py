#importing packages
import numpy as np
import pandas as pd
import tqdm

# importing dataframes
# thank you AGK: https://www.kaggle.com/akosciansky/how-to-import-large-csv-files-and-save-efficiently

# file path
path = '../input/'

# datatypes to reduce memory
# 'GameKey', 'GSISID', and 'PlayID' as 'str' to create unique ID
dtypes = {'Season_Year': 'int16',
         'GameKey': 'str',
         'PlayID': 'str',
         'GSISID': 'str',
         'Time': 'str',
         'x': 'float32',
         'y': 'float32',
         'dis': 'float32',
         'o': 'float32',
         'dir': 'float32',
         'Event': 'str'}

col_names = list(dtypes.keys())

# name of data files
data_files = [
    'NGS-2016-post.csv',
    'NGS-2016-pre.csv',
    'NGS-2016-reg-wk1-6.csv',
    'NGS-2016-reg-wk7-12.csv',
    'NGS-2016-reg-wk13-17.csv',
    'NGS-2017-post.csv',
    'NGS-2017-pre.csv',
    'NGS-2017-reg-wk1-6.csv',
    'NGS-2017-reg-wk7-12.csv',
    'NGS-2017-reg-wk13-17.csv',
    ]

# for loop that reads csv and appends dataframe to 'df'
# tqdm used to show progress bar of for loop
df_list = []

for i in tqdm.tqdm(data_files):
    ngs_combined = pd.read_csv(f'{path}'+i, usecols=col_names,dtype=dtypes)
    
    df_list.append(ngs_combined)

# creating unique_id to match to other datasets
ngs_combined['unique_id'] = (
    ngs_combined['GameKey']
    + '_'
    + ngs_combined['GSISID']
    + '_'
    + ngs_combined['PlayID']
    )
    
# exporting ngs_combined as csv
ngs_combined.to_csv('ngs_combined.csv', index=False)