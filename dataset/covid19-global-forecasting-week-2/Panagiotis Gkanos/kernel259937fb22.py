# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:40:49 2020

@author: Panagiotis Gkanos
"""
import tensorflow as tf
tf.compat.v1.set_random_seed(2)
import os
os.environ['KERAS_BACKEND']='tensorflow'

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn import metrics
from sklearn import preprocessing
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import datetime


def weird_division(a,b):
    if b == 0:
        return 0
    return a / b




def calculated_dr_features(df):
  '''
  Function: Calculates death rate features (death rate and active death rate)
  '''
  df['Death Rate']=""
  for i in range(df.shape[0]):
    a=df['Fatalities'][i]
    b=df['ConfirmedCases'][i]
    df['Death Rate'][i]=weird_division(a,b)
    
  return df

def calculate_diff_features(df,dates,keyword):
  '''
  Function: Calculates difference features like New Case etc

  '''
  df[keyword]=""
  for i in range(df.shape[0]):
    if ((i%(dates.shape[0]))==0):
      df[keyword][i]=0
    else:
      if keyword=='New Cases':
        df[keyword][i]=df['ConfirmedCases'][i]-df['ConfirmedCases'][i-1]
      elif keyword=='New Deaths':
        df[keyword][i]=df['Fatalities'][i]-df['Fatalities'][i-1]
      else:
        print('Wrong Keyword, try again!')
  return df


def calculate_pct_features(df,dates):

  '''
  Function: Calculates percentage features like New Case percentage etc
  '''
  df['New Confirmed pct']=""
  df['New Deaths pct']=""
  
  
  for i in range(df.shape[0]):
    if ((i%(dates.shape[0]))==0):
      df['New Confirmed pct'][i]=0
      df['New Deaths pct'][i]= 0
      
    else:      
      df['New Confirmed pct'][i]=weird_division(df['New Cases'][i],df['New Cases'][i-1])
      df['New Deaths pct'][i]= weird_division(df['New Deaths'][i],df['New Deaths'][i-1])
  return df


def extract_dates_from_dataframe(df):
    # Input: Dataframe with dates (one of confirmed,death,recovered)
    # Output: List with dates
    dates=[]
    for i in df.columns[4:len(df.columns)]:
      day=datetime.datetime.strptime(i,'%m/%d/%y').date()
      string=day.strftime("%Y-%m-%d")
      dates.append(string)
    return dates





filepath="/kaggle/input/covid19-global-forecasting-week-2/train.csv"

data=pd.read_csv(filepath)
data=data.sort_values(by=['Country_Region','Province_State','Date'])
data=data.rename(columns={'Unnamed: 0':'Id'})
data = data.reset_index(drop=True)#, inplace=True)
data['Id'] = data.index + 1
c=data

dates=c['Date']

c=calculated_dr_features(c)
c=calculate_diff_features(c,dates,'New Cases')
c=calculate_diff_features(c,dates,'New Deaths')
values={'Province/State': ""}
c.fillna(value=values,inplace=True)

for k in range (len(c)):
    if c['New Cases'][k]<0:
        c['New Cases'][k]= c['ConfirmedCases'][k]
for k in range (len(c)):
    if c['New Deaths'][k]<0:
        c['New Deaths'][k]=c['Fatalities'][k]
    
c=calculate_pct_features(c,dates)

c['BreakOut']=''
for p in range(len(c)):
    if c['ConfirmedCases'][p]==0:
        c['BreakOut'][p]=0
    else:
        c['BreakOut'][p]= c['BreakOut'][p-1]+1
 
c=c.rename(columns={'Province_State':'Province/State','Country_Region':'Country/Region','ConfirmedCases':'Confirmed','Fatalities':'Deaths'})   
count=0
for i in range(len(c)):
    if c["Date"][i]=="2020-01-22":
        count+=1
print(count)                
        
        
        
        
        
        
        
def weird_division(a,b):
    if b == 0:
        return 0
    return a / b

def plot_loss(history):
    train_loss=history.history['loss']
    #val_loss=history.history['val_loss']
    x=list(range(1,len(train_loss)+1))
    #plt.plot(x,val_loss,color='red',label='validation loss')
    plt.plot(x,train_loss,label='training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.show()

def convert_to_categorical(df,keyword):
  '''
  Converts categorical data to int format
  '''
  df[keyword]=df[keyword].astype('category',)
  df[keyword]=df[keyword].cat.codes
  return df

def convert_to_categorical_dates(df,keyword):
  '''
  Converts categorical data to int format
  '''
  df[keyword]=df[keyword].astype('category',)
  df[keyword]=df[keyword].cat.codes+57
  return df

#get the data and store them to a data frame
data=c

#fix the sort format of the data to match kaggle's dataset
data=data.sort_values(by=['Country/Region','Province/State','Date'])
data = data.reset_index(drop=True)#, inplace=True)
data['Id'] = data.index + 1


#convert text data to categorical
data=convert_to_categorical(data,'Province/State')
data=convert_to_categorical(data,'Country/Region')
data=convert_to_categorical(data,'Date')


#Create the final dataFrame with our features
W=data[['Province/State','Country/Region','Date','Confirmed','Deaths','Death Rate','New Cases','New Deaths','New Confirmed pct','New Deaths pct','BreakOut']]

'''
#normalize the data min-max  
tmp = W.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(tmp)
W = pd.DataFrame(x_scaled)    

#standarize data
scaler = StandardScaler()
scaler.fit(W)   
W = scaler.transform(W)
'''

#get the labels
#shift the labels with the window 
max_date =W['Date'].max()+1
labels=data[['Date','Confirmed','Deaths']]
old = 0
new = max_date
t_window = 3
for i in range(1,295):#CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    new=max_date*i
    labels[old:new] = labels[old:new].shift(-t_window,fill_value=0)
    old=new

Y=labels.loc[labels['Date'] != 0] #delete last day that we dont know the label 
Y = Y.reset_index(drop=True)
Y=Y[['Confirmed','Deaths']]
Y = np.array(Y)


#Seperate our Data in lists per Country
# T contains lists with the data of each country
W = np.array(W)
T = []
old = 0
new = 0

for i in range(1,295):#CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    new = max_date*i
    T[old:new] = [W[old:new]]
    old = new



#split our data in windows 
X = []
for i in range(len(T)):
    pi=0
    for k in range(max_date): #66
        if len(T[i][k:-1])<t_window:
            break
        else:
            X.append(T[i][k:k+t_window])
            pi=pi+1
                       


#merge test train data
Y_train = []
Y_test = []
X_train = []
X_test = []

old = 0
new = max_date-t_window-10
new_t = 10
for i in range(1,295):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    new=(max_date-t_window-10)*i
    X_train[old:new]=X[old:new]
    X_test[old:new_t] = X[new:(new+new_t)]
    Y_train[old:new] = Y[old:new]
    Y_test[old:new_t] = Y[new:(new+new_t)]
    old=new


# Convert lists to arrays too feed them to the NN
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

            
#LSTM model      
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(3,11)))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')


# Train model
history = model.fit(X_train,Y_train, epochs=700, verbose=1 , batch_size = 67)
plot_loss(history)
 

#Function to calulate logarithmic sq error         
def rmse(pred,val):
  logpred=abs(np.log(abs(pred)+1))
  logval=np.log(val+1)
  rmse=np.sqrt(np.sum(logpred-logval)/val.shape[0])
  return np.mean(rmse)


#model evaluation
pred=np.round(model.predict(X_test))
score=rmse(pred,Y_test)
print("RMSE : {}".format(score))

act1 = np.sqrt(mean_squared_log_error( Y_test,abs(pred)))

te = pred.clip(min=0)
te = abs(te)
act2 = np.sqrt(mean_squared_log_error( Y_test, te))
print("RMSE : {}".format(act2))

#kaggle unpredicted data

filepath="/kaggle/input/covid19-global-forecasting-week-2/test.csv"

k_test = pd.read_csv(filepath)

k_test['Confirmed']=''
k_test['Deaths']=''
k_test['Death Rate']=''
k_test['New Cases']=''
k_test['New Deaths']=''
k_test['New Confirmed pct']=''
k_test['New Deaths pct'] =''
k_test['BreakOut']=''

pos_k = 0
overlap = 13
pos_d = max_date-overlap
for i in range(1,295):
    k_test['Confirmed'][pos_k:pos_k+overlap] =  data['Confirmed'][pos_d:pos_d+overlap]
    k_test['Deaths'][pos_k:pos_k+overlap] = data['Deaths'][pos_d:pos_d+overlap]
    k_test['Death Rate'][pos_k:pos_k+overlap] = data['Death Rate'][pos_d:pos_d+overlap]
    k_test['New Cases'][pos_k:pos_k+overlap] = data['New Cases'][pos_d:pos_d+overlap]
    k_test['New Deaths'][pos_k:pos_k+overlap] = data['New Deaths'][pos_d:pos_d+overlap]
    k_test['New Confirmed pct'][pos_k:pos_k+overlap] = data['New Confirmed pct'][pos_d:pos_d+overlap]
    k_test['New Deaths pct'][pos_k:pos_k+overlap]  = data['New Deaths pct'][pos_d:pos_d+overlap]
    k_test['BreakOut'][pos_k:pos_k+overlap]= data['BreakOut'][pos_d:pos_d+overlap]
  
    pos_k=pos_k + 43
    pos_d = pos_d + max_date
  

   
k_test = k_test[['Province_State','Country_Region','Date','Confirmed','Deaths','Death Rate','New Cases','New Deaths','New Confirmed pct','New Deaths pct','BreakOut']]    

k_test=convert_to_categorical(k_test,'Province_State')
k_test=convert_to_categorical(k_test,'Country_Region')
k_test=convert_to_categorical_dates(k_test,'Date')
k_test = np.asarray(k_test)

t= 42
inp=[]


for i in range(0,294):
    for k in range(overlap-3,40):
        z=i*43+k
        inp = [k_test[z:z+3]]
        tmp = model.predict(np.array(inp))
        tmp = np.round(tmp.clip(min=0))
        k_test[z+3,3] = tmp[0,0]
        k_test[z+3,4] = tmp[0,1]
        k_test[z+3,5] = weird_division(k_test[z+3,4],k_test[z+2,3]) #death rate
        k_test[z+3,6] = k_test[z+3,3] - k_test[z+2,3] #new case confirmed
        k_test[z+3,7] = k_test[z+3,4] - k_test[z+2,4] #new case dead
        k_test[z+3,8] = weird_division(k_test[z+3,5],k_test[z+2,5]) #new pct confirmed
        k_test[z+3,9] = weird_division(k_test[z+3,6],k_test[z+2,6]) #new pct dead
        
        if k_test[z+2,-1] ==0 and k_test[z+3,3]!=0:
                k_test[z+3,-1] = 1
        else:
                k_test[z+3,-1] = k_test[z+2,-1] +1
        k_test[z+3] =  k_test[z+3].clip(min=0)
        


#make sub file

filepath="/kaggle/input/covid19-global-forecasting-week-2/submission.csv"


sub = pd.read_csv(filepath)
sub['ConfirmedCases'] = k_test[:,3].astype(int)
sub['Fatalities'] = k_test[:,4].astype(int)
sub.to_csv("submission.csv",index = False)
