# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

samp = 100  # Read every 'samp' row
with open('../input/train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    columns = spamreader.__next__()
    train_samp = []
    for i, row in enumerate(spamreader):
        if i % samp == 0:
            train_samp.append(row)
        if i % 1000000 == 0:
            print(i)
csvfile.close()
train_samp = pd.DataFrame(train_samp, columns=columns)
train_samp.index = train_samp.user_id
del train_samp['user_id']
print(train_samp)
train_samp.to_csv('train_sparse_100.csv')