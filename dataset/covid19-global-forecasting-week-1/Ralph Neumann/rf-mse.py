import numpy as np
import pandas as pd

from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# /kaggle/input/data/covid19-global-forecasting-week-1/test.csv'

src_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
src_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
sub_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

src_train.head()

src_train['Date'] = src_train['Date'].apply(lambda x: int(x.replace('-','')))
src_test['Date'] = src_test['Date'].apply(lambda x: int(x.replace('-','')))

le = preprocessing.LabelEncoder()
le.fit(src_train['Country/Region'].astype(str))

src_train['Country/Region'] = le.transform(src_train['Country/Region'])
src_test['Country/Region'] = le.transform(src_test['Country/Region'])

src_train.head()

X_train = pd.DataFrame(src_train.iloc[:,[2,3,4,5]])
y1_train = pd.DataFrame(src_train.iloc[:,[6]])
y2_train = pd.DataFrame(src_train.iloc[:,[7]])
X_test = pd.DataFrame(src_test.iloc[:,[2,3,4,5]])

X_train.head()

def ml_algo(algorithm, X_train, y_train, cv):
    
    # One Pass
    model = algorithm.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = cross_val_predict(algorithm, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv)
    # Cross-validation accuracy metric
    scores = cross_val_score(algorithm,X_train,y_train, cv=cv)
    acc_cv = round(scores.mean()* 100, 2)
    
    #R-2 square value
    R2_value = round(metrics.r2_score(y_train, train_pred) * 100 ,2)

    RMSE = np.sqrt(metrics.mean_squared_error(y_train, train_pred))
    
    return train_pred, acc, acc_cv, R2_value,RMSE
	
rf_pred,rf_acc,rf_acc_cv,rf_r2_value,rf_RMSE = ml_algo(RandomForestClassifier(criterion="entropy"),X_train,y1_train,10)

print("Accuracy of Random Forest Classifier : ",rf_acc)
print("Cross validation accuracy with cv=10 : ",rf_acc_cv)
print ("R-Square Accuracy : ", rf_r2_value)
print("Root Mean Square Error : ", rf_RMSE)

rf = RandomForestClassifier(criterion="entropy")
rf.fit(X_train,y1_train)

ConfirmedCases = rf.predict(X_test)

rf_fatality = RandomForestClassifier(criterion="entropy")
rf_fatality.fit(X_train,y2_train)

Fatalities = rf_fatality.predict(X_test)

sub_data.head()

sub_df = pd.DataFrame()
sub_df['ForecastId'] = sub_data['ForecastId']
sub_df['ConfirmedCases'] = ConfirmedCases
sub_df['Fatalities'] = Fatalities

sub_df.to_csv('submission.csv',index=False)