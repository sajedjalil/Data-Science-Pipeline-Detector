# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.linear_model import LogisticRegression


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
 #from subprocess import check_output
 #print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Read data
df_train = pd.read_csv("../input/train.csv")
df_test  = pd.read_csv("../input/test.csv")

# Preview the data
# Each row: insurance claim
# predict the value for the 'loss' column.
# 'cat' are categorical
# 'cont' are continuous

print(df_test.head())
#print(df_train.shape)  # (188318, 132)  Zeile x Spalte
#print(df_test['cont1'].mean())
#print(df_test['cont1'].max())
print(df_test['cat116'].head()+df_test['cat115'].head())

# Shuffle Pandas data frame
#import sklearn.utils
#df = sklearn.utils.shuffle(df)
#print('\n\ndf: {0}'.format(df))