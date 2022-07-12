import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#score
from sklearn.metrics import mean_squared_log_error

def rmsle(y, y_pred):     

    return np.sqrt(mean_squared_log_error( y, y_pred ))
    
        
# Load train and test.

train=pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
train.set_index('Id',inplace=True)
test=pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
test.set_index('ForecastId',inplace=True)

x_train=train.iloc[:,0:3]
y_train=train.iloc[:,3:5]
x_test=test

#Feature engineering
x_train['Date'] = pd.to_datetime(x_train.Date)
x_train['Date']=x_train['Date'].map(dt.datetime.toordinal)
x_test['Date'] = pd.to_datetime(x_test.Date)
x_test['Date']=x_test['Date'].map(dt.datetime.toordinal)

#print(x_train.dtypes,'\n \n',x_test.dtypes)
x_train=pd.get_dummies(data=x_train,columns=['Province_State','Country_Region'])
x_test=pd.get_dummies(data=x_test,columns=['Province_State','Country_Region'])
print('Shape Size ',x_train.shape,'Shape Test ',x_test.shape)
print(x_train.dtypes,'\n \n',x_test.dtypes)


#Train
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition, datasets

steps = [\
         ('scaler', StandardScaler()),
         ('Model',  MultiOutputRegressor(DecisionTreeRegressor()))      
        ]
pipeline = Pipeline(steps) 
parameteres = {\
               'Model__estimator__criterion':['mse','friedman_mse']

              }

my_scorer = make_scorer(rmsle, greater_is_better=False)

grid = GridSearchCV(pipeline, param_grid=parameteres, 
#                    scoring=my_scorer,
                    cv=5,n_jobs=-1)

grid.fit(x_train, y_train)
y_train_predict=grid.predict(x_train)
#print("score = %3.2f" %(rmsle(y_train, y_train_predict)))
print("score = %3.2f" %grid.score(x_train, y_train))
print(grid.best_params_)

y_test_predict=grid.predict(x_test)

sub=pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')
sub['ConfirmedCases']=y_test_predict[:,0]
sub['Fatalities']=y_test_predict[:,1]
sub.to_csv(r'submission.csv',index=False)









