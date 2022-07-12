# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train_2016.csv')
prop_df = pd.read_csv('../input/properties_2016.csv')

###### remove columns with >90% null
prop_frac = prop_df.isnull().sum()/prop_df.shape[0]
prop_frac = prop_frac.reset_index()
prop_frac.columns = ["column_name", "null_frac"]
null_cols = prop_frac.loc[prop_frac.null_frac>0.9, "column_name"].tolist()
prop_df.drop(null_cols, axis=1, inplace=True)

obj_cols = [col for col in prop_df.columns if prop_df[col].dtypes=='object']
prop_df.loc[:,obj_cols] = prop_df[obj_cols].astype('str')
dense_cols = [col for col in prop_df.columns if prop_df[col].dtypes=='float64']
prop_df.loc[:,dense_cols] = prop_df[dense_cols].astype('float32')
prop_df.loc[:,dense_cols] = prop_df[dense_cols].fillna(prop_df[dense_cols].mean())

for col in obj_cols:
    assert 'null' not in prop_df[col].values
    
prop_df.loc[:,obj_cols].fillna('null', inplace=True)

prop_df.drop(obj_cols, axis=1, inplace=True)

train_data = pd.merge(left=train_df, right=prop_df, on='parcelid', how='left')

print(train_data.shape)