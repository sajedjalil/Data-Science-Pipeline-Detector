# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
#from sklearn.metrics import mean_squared_error
#import seaborn as sns
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
dt = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df.drop(['casual','registered'],axis = 1,inplace = True)

#
le = LabelEncoder()
#
df['datetime']=(le.fit_transform(df['datetime']))
dt['datetime']=(le.fit_transform(dt['datetime']))


X = df.iloc[:,0:9]
y = df.iloc[:,9]

x_test = dt.iloc[:,0:9]

rfr = RandomForestRegressor()
rfr.fit(X,y)
y_pred = rfr.predict(x_test)
#print(y_pred)
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
#print(sub_df.shape)
datasets=pd.concat([sub_df['datetime'],pred],axis=1)
datasets.columns=['datetime','count']
datasets.to_csv('submission.csv',index=False)
