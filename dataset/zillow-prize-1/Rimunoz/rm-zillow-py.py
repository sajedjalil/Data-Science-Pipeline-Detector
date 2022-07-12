# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#EXPLORE TRAIN DATASET
df_train = pd.read_csv("../input/train_2016.csv",parse_dates=['transactiondate'])

print(df_train.head(5))
print(df_train.describe())

#EXPLORE PROPERTIES DATASET
df_properties = pd.read_csv("../input/properties_2016.csv", dtype={'hashottuborspa':np.str,'propertycountylandusecode':np.str,'propertyzoningdesc':np.str,'fireplaceflag':np.str,'taxdelinquencyflag':np.str})

print(df_properties.head(5))
df_properties.hist()

#MERGE DATASET
df = df_train.merge(df_properties, how='left', on='parcelid')

print(df.count())






