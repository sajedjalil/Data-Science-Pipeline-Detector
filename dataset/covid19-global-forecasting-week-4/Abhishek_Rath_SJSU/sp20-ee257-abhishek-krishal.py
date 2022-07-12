# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:43.270912Z","start_time":"2020-04-15T18:48:43.251910Z"}}

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.437097Z","start_time":"2020-04-15T18:48:43.278915Z"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.471107Z","start_time":"2020-04-15T18:48:46.443106Z"}}
df_train=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
df_test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
df_sub=pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

print(df_train.shape)
print(df_test.shape)
print(df_sub.shape)

# %% [markdown]
# ### EDA Train Data

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.626095Z","start_time":"2020-04-15T18:48:46.595093Z"}}
df_train.head()

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.646102Z","start_time":"2020-04-15T18:48:46.632093Z"}}
print(f"Unique Countries: {len(df_train.Country_Region.unique())}")

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.683116Z","start_time":"2020-04-15T18:48:46.657099Z"}}
train_dates=list(df_train.Date.unique())
latest_date=df_train.Date.max()
print(f"Period : {len(df_train.Date.unique())} days")
print(f"From : {df_train.Date.min()} To : {df_train.Date.max()}")

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.699099Z","start_time":"2020-04-15T18:48:46.690098Z"}}
print(f"Unique Regions: {df_train.shape[0]/len(df_train.Date.unique())}")

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.727101Z","start_time":"2020-04-15T18:48:46.707099Z"}}
df_train.Country_Region.value_counts()

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.851118Z","start_time":"2020-04-15T18:48:46.753103Z"}}
df_train["UniqueRegion"]=df_train.Country_Region
df_train.UniqueRegion[df_train.Province_State.isna()==False]=df_train.Province_State+" , "+df_train.Country_Region

region_list=df_train.UniqueRegion.unique()
print(f"Total unique regions are : {len(region_list)}")
df_train[df_train.Province_State.isna()==False]

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:46.893112Z","start_time":"2020-04-15T18:48:46.857110Z"}}
df_train.drop(labels=["Id","Province_State","Country_Region"], axis=1, inplace=True)
df_train

# %% [markdown]
# We will add one more column, Delta (Growth Factor) which is the ratio of confirmed cases on one day to that of the previous day.

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:58.515800Z","start_time":"2020-04-15T18:48:46.902119Z"}}
df_train["Delta"]=1.0
df_train["NewCases"]=0.0
final_train=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","NewCases","UniqueRegion", "Delta"])

for region in region_list:
    df_temp=df_train[df_train.UniqueRegion==region].reset_index()
    size_train=df_temp.shape[0]
    
    df_temp.NewCases[0]=df_temp.ConfirmedCases[1]
    for i in range(1,df_temp.shape[0]):
        df_temp.NewCases[i]=df_temp.ConfirmedCases[i]-df_temp.ConfirmedCases[i-1]
        if(df_temp.ConfirmedCases[i-1]>0):
            df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]
            
    df_temp=df_temp[["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"]]
    final_train=pd.concat([final_train,df_temp], ignore_index=True)
final_train.shape

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:48:58.576807Z","start_time":"2020-04-15T18:48:58.534808Z"}}
latest_train=final_train[final_train.Date==latest_date]
latest_train.head()

# %% [markdown]
# ### Define a function to plot how confirmed cases, fatalities and Delta changes with time

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:49:57.240039Z","start_time":"2020-04-15T18:49:55.065875Z"}}
score_list=[]
for region in region_list:
    df_temp=final_train[final_train.UniqueRegion==region]
    X=np.array(df_temp.ConfirmedCases).reshape(-1,1)
    Y=df_temp.Fatalities
    model=LinearRegression()
    model.fit(X,Y)
    score_list.append(model.score(X,Y))
score_df=pd.DataFrame({"Region":region_list,"Score":score_list})
print(f"Average R2 score between Fatality and Confirmed Cases is :{score_df.Score.mean()}")

plt.figure(figsize=(10,6))    
plt.title("Distribution of R2 score between Confirmed Cases and Fatality")
sns.distplot(score_df.Score)
plt.show()

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:49:57.320070Z","start_time":"2020-04-15T18:49:57.253061Z"}}
less_than_50=score_df[score_df.Score<0.5].Region.unique()
print(f"Number of countries where r2 score<0.50 : {len(less_than_50)}")
latest_train[latest_train.UniqueRegion.isin(less_than_50)]

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:50:33.193517Z","start_time":"2020-04-15T18:50:16.455575Z"},"scrolled":false}

%%time
reg_score_list=[]
period=[]
reg=[]
for n in range(3,10):
    for region in region_list:
        df_temp=final_train[final_train.UniqueRegion==region]
        df_temp=df_temp.tail(n).reset_index()
        date=np.arange(1,n+1)
        model=LinearRegression()
        X=date.reshape(-1,1)
        Y=df_temp.Delta
        model.fit(X,Y)
        reg.append(region)
        reg_score_list.append(model.score(X,Y))
        period.append(n)
score_df=pd.DataFrame({"Region":reg,"Score":reg_score_list, "Period":period})

# %% [markdown]
# Observation:
# * Generally average R2 score is better when N=3
# * For some regions, R2 score is better when N is higer

# %% [markdown]
# ### Finding Best N for regions

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:50:34.228593Z","start_time":"2020-04-15T18:50:33.198523Z"}}

n_list=[]
for reg in region_list:
    temp_score_df=score_df[score_df.Region==reg]
    if temp_score_df.Score.max()==1:
        n_list.append(3)
    else:
        n_list.append(temp_score_df.Period[temp_score_df.Score==temp_score_df.Score.max()].median())
best_n_df=pd.DataFrame({"Region":region_list,"N":n_list})
sns.countplot(best_n_df.N)


# %% [markdown] {"ExecuteTime":{"end_time":"2020-04-15T17:33:38.245393Z","start_time":"2020-04-15T17:33:38.240390Z"}}
# * Observation Linear Regression can be used to predict Delta for the test data however the model would be underfitting.
# * Polynomial Regression or LSTM would do a much better job in prediction

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:50:34.262598Z","start_time":"2020-04-15T18:50:34.233594Z"}}
df_test.head()

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T19:15:47.277304Z","start_time":"2020-04-15T19:15:47.243302Z"}}
print(f"Unique Countries: {len(df_test.Country_Region.unique())}")
test_dates=list(df_test.Date.unique())
size_test=len(df_test.Date.unique())
print(f"Period : {len(df_test.Date.unique())} days")
print(f"From : {df_test.Date.min()} To : {df_test.Date.max()}")
print(f"Unique Regions: {df_test.shape[0]/len(df_test.Date.unique())}")

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:50:34.329607Z","start_time":"2020-04-15T18:50:34.294601Z"}}
df_test["UniqueRegion"]=df_test.Country_Region
df_test.UniqueRegion[df_test.Province_State.isna()==False]=df_test.Province_State+" , "+df_test.Country_Region

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:50:34.354609Z","start_time":"2020-04-15T18:50:34.333604Z"}}
df_test.drop(labels=["ForecastId","Province_State","Country_Region"], axis=1, inplace=True)
df_test["ConfirmedCases"]=0
df_test["Fatalities"]=0
df_test["NewCases"]=0
df_test["Delta"]=0

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T18:50:34.382607Z","start_time":"2020-04-15T18:50:34.358608Z"}}
final_test=df_test[["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"]]
app_test=final_test[final_test.Date>latest_date]
app_test.shape

# %% [markdown] {"ExecuteTime":{"end_time":"2020-04-15T20:14:24.229708Z","start_time":"2020-04-15T20:14:24.225707Z"}}
# ## Predicting using Linear Regression of Delta

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T20:53:21.298702Z","start_time":"2020-04-15T20:53:05.013812Z"}}

df_pred=pd.DataFrame(columns=["ConfirmedCases","Fatalities"])
df_traintest=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"])

for region in region_list:
    df_temp=final_train[final_train.UniqueRegion==region].reset_index()
    
    #number of days for delta trend
    n=int(best_n_df[best_n_df.Region==region].N.sum()) 
    #Delta for the period
    delta_list=np.array(df_temp.tail(n).Delta).reshape(-1,1)
    #Morality rate as on last availabe date
    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()
    
    scaler=MinMaxScaler()
    X=np.arange(1,n+1).reshape(-1,1)
    Y=scaler.fit_transform(delta_list) 
    model=LinearRegression()
    model.fit(X,Y)
    
    df_test_app=app_test[app_test.UniqueRegion==region]
    df_temp=pd.concat([df_temp,df_test_app])
    df_temp=df_temp.reset_index()
    
    for i in range (size_train, df_temp.shape[0]):
        n=n+1        
        df_temp.Delta[i]=max(1,scaler.inverse_transform(model.predict(np.array([n]).reshape(-1,1))))
        df_temp.ConfirmedCases[i]=round(df_temp.ConfirmedCases[i-1]*df_temp.Delta[i],0)
        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)
        df_temp.NewCases[i]=df_temp.ConfirmedCases[i]-df_temp.ConfirmedCases[i-1]
        
    df_traintest=pd.concat([df_traintest,df_temp],ignore_index=True)
    
    df_temp=df_temp.iloc[-size_test:,:]
    df_temp=df_temp[["ConfirmedCases","Fatalities"]]
    df_pred=pd.concat([df_pred,df_temp], ignore_index=True)


# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T20:53:50.428207Z","start_time":"2020-04-15T20:53:45.345781Z"}}
def prediction_plotter(r_name):
    pred_df=df_traintest[df_traintest.UniqueRegion==r_name]
    train_df=final_train[final_train.UniqueRegion==r_name]
    plt.figure(figsize=(10,6))
    sns.lineplot('Date','ConfirmedCases',data=pred_df, color='r', label="Predicted Cases")
    sns.lineplot('Date','ConfirmedCases',data=train_df, color='g', label="Actual Cases")
    plt.show()

# %% [code]
prediction_plotter("Germany")

# %% [code]
prediction_plotter("Pakistan")

# %% [markdown]
# ## Prediction Where new Cases follows Linear Regression

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T21:09:16.144782Z","start_time":"2020-04-15T21:09:00.959475Z"}}

df_pred=pd.DataFrame(columns=["ConfirmedCases","Fatalities"])
df_traintest=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"])

for region in region_list:
    df_temp=final_train[final_train.UniqueRegion==region].reset_index()
    
    #number of days for delta trend
    n=10 
    #Delta for the period
    NewCasesList=df_temp.tail(n).NewCases 
    #Morality rate as on last availabe date
    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()
    
    X=np.arange(1,n+1).reshape(-1,1)
    Y=NewCasesList
    model=LinearRegression()
    model.fit(X,Y)
    
    df_test_app=app_test[app_test.UniqueRegion==region]
    df_temp=pd.concat([df_temp,df_test_app])
    df_temp=df_temp.reset_index()
    
    for i in range (size_train, df_temp.shape[0]):
        n=n+1        
        df_temp.NewCases[i]=round(max(0,model.predict(np.array([n]).reshape(-1,1))[0]),0)
        df_temp.ConfirmedCases[i]=df_temp.ConfirmedCases[i-1]+df_temp.NewCases[i]
        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)
        df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]
        
    df_traintest=pd.concat([df_traintest,df_temp],ignore_index=True)
    
    df_temp=df_temp.iloc[-size_test:,:]
    df_temp=df_temp[["ConfirmedCases","Fatalities"]]
    df_pred=pd.concat([df_pred,df_temp], ignore_index=True)
df_pred.shape


# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T21:10:02.461045Z","start_time":"2020-04-15T21:09:57.181053Z"}}
prediction_plotter("New York , US")

# %% [code]
prediction_plotter("Korea, South")

# %% [markdown]
# ## Prediction when confirmed cases is in polinomial regression

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T21:44:11.051128Z","start_time":"2020-04-15T21:44:00.550340Z"}}
#"""
df_pred=pd.DataFrame(columns=["ConfirmedCases","Fatalities"])
df_traintest=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"])

for region in region_list:
    df_temp=final_train[final_train.UniqueRegion==region].reset_index()
    
    #number of days for delta trend
    n=7
    #Delta for the period
    ConfirmedCasesList=df_temp.tail(n).ConfirmedCases 
    #Morality rate as on last availabe date
    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()
    polynom=PolynomialFeatures(degree=2)
    X=polynom.fit_transform(np.arange(1,n+1).reshape(-1,1))
    Y=ConfirmedCasesList
    model=LinearRegression()
    model.fit(X,Y)
    
    df_test_app=app_test[app_test.UniqueRegion==region]
    df_temp=pd.concat([df_temp,df_test_app])
    df_temp=df_temp.reset_index()
    
    for i in range (size_train, df_temp.shape[0]):
        n=n+1        
        pred=round(model.predict(polynom.fit_transform(np.array(n).reshape(-1,1)))[0],0)
        df_temp.ConfirmedCases[i]=max(df_temp.ConfirmedCases[i-1],pred)
        df_temp.NewCases[i]=df_temp.ConfirmedCases[i]+df_temp.ConfirmedCases[i-1]
        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)
        df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]
        
    df_traintest=pd.concat([df_traintest,df_temp],ignore_index=True)
    
    df_temp=df_temp.iloc[-size_test:,:]
    df_temp=df_temp[["ConfirmedCases","Fatalities"]]
    df_pred=pd.concat([df_pred,df_temp], ignore_index=True)
df_pred.shape
#"""

# %% [code] {"ExecuteTime":{"end_time":"2020-04-15T21:44:31.976411Z","start_time":"2020-04-15T21:44:29.003189Z"}}
prediction_plotter("India")

# %% [code]
prediction_plotter("New York , US")

# %% [code]
df_sub.ConfirmedCases=df_pred.ConfirmedCases
df_sub.Fatalities=df_pred.Fatalities
#df_sub.to_csv("submission.csv",index=None)