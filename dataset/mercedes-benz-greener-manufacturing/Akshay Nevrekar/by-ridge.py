# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy as sp

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge



train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

ID=test['ID']


y_train=train['y']   #storing target variable in y 
train=train.drop(['y'],1)  #removing target from train dataset 


#combine train and test dataset 
df=[train,test]
df=pd.concat(df, axis=0)
print(df.head())

df=df.drop(['ID','X4'],1)

df['X0']=df['X0'].astype('category')
df['X1']=df['X1'].astype('category')
df['X2']=df['X2'].astype('category')
df['X3']=df['X3'].astype('category')
#df['X4']=df['X4'].astype('category')
df['X5']=df['X5'].astype('category')
df['X6']=df['X6'].astype('category')
df['X8']=df['X8'].astype('category')

#convert categorical features into dummy variables
df=pd.get_dummies(df,columns=['X0','X1','X2','X3','X5','X6','X8'], drop_first=True)

#split tain and test dataset using index
train=df[0:4209]

test=df[4209:]


#Building a model with Linear Regression
reg= Ridge(alpha=100, random_state=42)

reg.fit(train,y_train)

y=reg.predict(test)


#creating submission file
#ID=test['ID']
df=[ID,y]
df=pd.DataFrame(df)
df=df.transpose()
df.columns.values[1] = 'y'  
df['ID']=df['ID'].astype('Int32')
#df['y']=df['y'].astype('Float64')
#op=pd.concat(op, axis=1)

df.to_csv("output_ridge1.csv", index=False)