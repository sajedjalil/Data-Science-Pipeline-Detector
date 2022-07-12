import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

path ='../input/hubber-models/'

train = pd.read_csv(path + 'train.csv',usecols = ['train_id','item_description'],index_col=['train_id'])
test = pd.read_csv('../input/mercari-price-suggestion-challenge/test.tsv',sep='\t'
    ,usecols=['test_id','item_description'],index_col=['test_id'])
print(train.head())