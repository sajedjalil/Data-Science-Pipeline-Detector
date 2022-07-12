# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
sub = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')

train.rename(columns={'Id': 'id',
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'Date': 'date', 
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                    }, inplace=True)

test.rename(columns={'ForecastId': 'id',
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'Date': 'date', 
                     }, inplace=True)

print(train.info())



data_train = train.sort_values(by=['country','state','date'],ascending=[True,True,True])
print(data_train.tail())

data_test = test.sort_values(by=['country','state','date'],ascending=[True,True,True])
print(data_test.tail())

print(data_train['long'].mean())
data_train["lat"]  = data_train["lat"].fillna(26.33)
data_train["long"]  = data_train["long"].fillna(5.03)
data_test["lat"]  = test["lat"].fillna(12.5211)
data_test["long"]  = test["long"].fillna(69.9683)
data_train['date'] = data_train['date'].apply(lambda x: x.replace("-",""))
data_train['date'] = data_train['date'].astype(int)

data_test['date'] = data_test['date'].apply(lambda x: x.replace("-",""))
data_test['date'] = data_test['date'].astype(int)

corr_matrix = data_train.corr()
print(corr_matrix['confirmed'].sort_values(ascending = False))

m=0
n =59
a=0
b=43

for i in range(284):
    
    if i==0:
          train_data = data_train.iloc[m:n,3:6]
          val = data_test.iloc[a:b,3:6]
         
          confirmed = data_train.iloc[m:n,6]
          deaths = data_train.iloc[m:n,7]
         
          #model_c = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
          model_c = RandomForestRegressor(criterion = 'mse',random_state=500)
          #model_d = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
          model_d=RandomForestRegressor(criterion = 'mse',random_state=500)
          model_c.fit(train_data,confirmed)
          model_d.fit(train_data,deaths)
    
          pred_c = model_c.predict(val)
          pred_d = model_d.predict(val)
          p_c = pd.DataFrame(pred_c)
          p_d = pd.DataFrame(pred_d)
          p_c.columns = ["confirmed"]  
          p_d.columns = ["death"]
          
          date_c= pd.to_datetime(train['date'][m:n])
          date_p = pd.to_datetime(test['date'][a:b])
          
          fig,ax =plt.subplots(2)
          #country = data_train.iloc[m:m+1)
          fig.suptitle(data_train.iloc[m,2])
          l_c1 = ax[0].plot(date_p,pred_c,label = 'Predicted_Cases',color ='red')
          l_c2 = ax[0].plot(date_c,confirmed,label = 'Confirmed_Cases', color ='blue')
          ax[0].set(xlabel="Date",ylabel="No_Of_Cases")
          ax[0].legend()
          l_d1 = ax[1].plot(date_p,pred_d,label = 'Predicted_Deaths' ,color ='green')
          l_d2 = ax[1].plot(date_c,deaths,label = 'Confirmed_Deaths', color ='orange')
          ax[1].set(xlabel="Date",ylabel="No_Of_Deaths")
          ax[1].legend()
          plt.show()
          
          a =b
          b+= 43 
          m = n
          n +=59
          
         

    else:
         train_data = data_train.iloc[m:n,3:6] 
         val = data_test.iloc[a:b,3:6]
         
         confirmed = data_train.iloc[m:n,6]
         deaths = data_train.iloc[m:n,7]
        
         #model_c =SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
         model_c = RandomForestRegressor(criterion = 'mse',random_state =500)
         #model_d =SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
         model_d =  RandomForestRegressor(criterion = 'mse',random_state=500)
         model_c.fit(train_data,confirmed)
         model_d.fit(train_data,deaths)
    
         pred_c = model_c.predict(val)
         pred_d = model_d.predict(val)
         t_c = pd.DataFrame(pred_c)
         t_d = pd.DataFrame(pred_d)
         p_c = p_c['confirmed'].append(t_c)
         p_d = p_d['death'].append(t_d)
         p_c.columns = ["confirmed"]  
         p_d.columns = ["death"]
         
         X= pd.to_datetime(train['date'][m:n])
         Y = pd.to_datetime(test['date'][a:b])
         
         fig,ax =plt.subplots(2)
         fig.suptitle(data_train.iloc[m,2])
         ax[0].plot(Y,pred_c,label = 'Predicted_Cases', color ='red')
         ax[0].plot(X,confirmed,label = 'Confirmed_Cases', color ='blue')
         ax[0].set(xlabel="Date",ylabel="No_Of_Cases")
         ax[0].legend()
         ax[1].plot(Y,pred_d,label = 'Predicted_Deaths', color ='green')
         ax[1].plot(X,deaths,label = 'Confirmed_Deaths', color ='orange')
         ax[1].set(xlabel="Date",ylabel="No_Of_Deaths")
         ax[1].legend()
         plt.show()
         m = n
         n +=59
         a =b
         b+= 43
         

for i in range(12212):
    sub.iloc[i,1] = p_c.iloc[i,0]
    sub.iloc[i,2] = p_d.iloc[i,0]                          

sub['ConfirmedCases'] = sub['ConfirmedCases'].astype(int)
sub['Fatalities'] = sub['Fatalities'].astype(int)

sub.to_csv('submission.csv',index = False)

