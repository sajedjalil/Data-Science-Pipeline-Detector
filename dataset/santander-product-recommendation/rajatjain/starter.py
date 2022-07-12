# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
pd.set_option('display.max_columns', 500)
# Any results you write to the current directory are saved as output.

df= pd.read_csv("../input/train_ver2.csv", nrows=10000)


no_col = ['conyuemp', 'ult_fec_cli_1t']
cols = [c for c in df.columns if c not in no_col]

df = df[cols].dropna()
# print(df[pd.isnull(df).any(axis=1)])
sns.pairplot(df.ix['ind_ahor_fin_ult1':'ind_recibo_ult1'])
plt.show()
print("sdssdsdcsd")
# print(df.head())