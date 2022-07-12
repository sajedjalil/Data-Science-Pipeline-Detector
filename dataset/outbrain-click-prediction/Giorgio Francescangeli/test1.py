# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# get titanic & test csv files as a DataFrame
#clicks_test_df = pd.read_csv("../input/clicks_test.csv")
clicks_train_df    = pd.read_csv("../input/clicks_train.csv")

# preview the data
#print(clicks_test_df.head())
#print(clicks_train_df.head(50))
ids = clicks_train_df.ad_id
#print(ids)
dist={}

for k in ids:
    if k in dist:
        dist[k]+=1
    else:
        dist[k]=1
print(dist)
# Any results you write to the current directory are saved as output.