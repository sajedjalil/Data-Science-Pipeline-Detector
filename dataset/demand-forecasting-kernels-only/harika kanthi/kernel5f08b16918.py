# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso,LassoCV,Ridge,RidgeCV,LinearRegression,SGDRegressor,ElasticNet,ElasticNetCV
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.feature_selection import RFE,RFECV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
# !pip install xgboost
from xgboost import XGBRegressor,XGBRFRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
test=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')
ans=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')
train.columns
df=train.copy()
df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['day']=df['date'].dt.day
df['dayofweek']=df['date'].dt.dayofweek
test['date']=pd.to_datetime(test['date'])
test['year']=test['date'].dt.year
test['month']=test['date'].dt.month
test['day']=test['date'].dt.day
test['dayofweek']=test['date'].dt.dayofweek
X=df[['store', 'item', 'year', 'month', 'dayofweek']]
y=df['sales']
testing=test[['store', 'item', 'year', 'month', 'dayofweek']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
print("Bagging Regression")
print("   ")
br=BaggingRegressor(n_estimators=10,n_jobs=-1)
br.fit(X_train,y_train)
print("R2 train score: ",br.score(X_train,y_train))
y_pred=br.predict(X_test)
print("R2 test score: ",metrics.r2_score(y_test,y_pred))
print("MSE: ",metrics.mean_squared_error(y_test,y_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("MAE: ",metrics.mean_absolute_error(y_test,y_pred))
pred=br.predict(testing)
my_submission=pd.DataFrame({'id':ans['id'],'sales':pred.astype(int)})
my_submission.to_csv('submission.csv', index=False)