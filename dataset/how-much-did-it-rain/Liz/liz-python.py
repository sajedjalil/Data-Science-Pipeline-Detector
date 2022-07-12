import pandas as pd
import os
import json

print(os.getcwd())
print(os.listdir(os.getcwd()))

print(os.listdir('..'))

print(os.listdir('../input'))

import zipfile
# json.loads(



infile="../input/train_2013.csv.zip"


zf = zipfile.ZipFile(infile)
df = pd.read_csv(zf.open('train_2013.csv'))

print(df.head())
# with zipfile.ZipFile(infile, 'r') as z:
#     f = z.open('train_2013.csv')
#     table = pd.read_csv('train_2013.csv')
# alldata = pd.read_csv(infile,  compression='gzip')

# alldata.head()