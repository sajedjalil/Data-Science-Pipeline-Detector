# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv', nrows=16000000)
df_train, df_test = train_test_split(df, test_size=0.25)

## Linear Model
from sklearn import linear_model
df_y = df_train['Demanda_uni_equil']
del(df_train['Demanda_uni_equil'])
del(df_train['Venta_uni_hoy'])

linreg = linear_model.LinearRegression()

linreg.fit(df_train, df_y)
print('Coefficients: \n', linreg.coef_)

df_y_test = df_test['Demanda_uni_equil']
del(df_test['Demanda_uni_equil'])
del(df_test['Venta_uni_hoy'])

print("Residual sum of squares: %.2f"
      % np.mean((linreg.predict(df_test) - df_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linreg.score(df_test, df_y_test))


#i = 0
#with open('../input/train.csv','r') as s:
#    st = csv.reader(s)
#    for row in st:
#        i+=1
#print(i)