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
from time import time

df = pd.read_csv('../input/clicks_train.csv', iterator=True, chunksize=100000)
adcounts = {}
nrows = 0
ti = time()
for data in df:
    for i, d in data.iterrows():
        adcounts[d.ad_id] = adcounts.get(d.ad_id, 0) + d.clicked
    nrows += data.shape[0]
    print('[{0:8.1f}s]Process {1} rows!'.format(time()-ti, nrows))
    break

f = open('adcounts.csv', 'w')
f.write('adid,count\n')
for adid, v in adcounts.items():
    f.write('{0},{1}\n'.format(adid, v))
f.close()
