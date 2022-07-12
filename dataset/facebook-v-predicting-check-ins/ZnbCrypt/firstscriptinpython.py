# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting figures
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#  Loading data into array
print('Reading train data')
df_train = pd.read_csv('../input/train.csv')
 
X = pd.DataFrame(df_train,columns=['row_id', 'x', 'y','time'])
Y = pd.DataFrame(df_train,columns=['place_id'])
XX=np.array(X);
X1 = XX[:,0]
YY=np.array(Y);
Y1=YY[:,0]
 
 
plt.scatter(X1,Y1 )
plt.ylabel('place_id')
plt.xlabel('X')
plt.show()

