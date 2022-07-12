# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:52:10 2020

@author: SpyrosV
"""
#imports
import warnings
warnings.filterwarnings("ignore")

from numpy.random import seed
seed(1)

import tensorflow as tf
tf.compat.v1.set_random_seed(2)
import os
os.environ['KERAS_BACKEND']='tensorflow'

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv1D,Conv2D,Flatten,TimeDistributed,BatchNormalization
from sklearn import metrics
from sklearn import preprocessing
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import datetime
from keras import regularizers
from keras import optimizers

#*****************************************************

#functions

def weird_division(a,b):
    if b == 0:
        return 1
    return a / b

def calculated_dr_features(df,dates):
  '''
  Function: Calculates death rate features (death rate and active death rate)
  '''
  df['Death Rate']=""
  for i in range(df.shape[0]):
    if ((i%(dates.shape[0]))==0):
        df['Death Rate'][i]=0
    else:
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
      df['New Confirmed pct'][i]=weird_division(df['ConfirmedCases'][i],df['ConfirmedCases'][i-1])
      df['New Deaths pct'][i]= weird_division(df['Fatalities'][i],df['Fatalities'][i-1])
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
  df[keyword]=df[keyword].cat.codes+72
  return df

def rmse(pred,val):
  x = pred.clip(min=0)
  x = abs(x)
  logpred=np.log(x+1)
  logval=np.log(val+1)
  rmse=np.sqrt(np.sum((logpred-logval)**2)/val.shape[0])
  return rmse


#******************************************************************************
  
#calculate features and fix data set
  
filepath='/kaggle/input/covid19-global-forecasting-week-4/train.csv'

data=pd.read_csv(filepath)
data=data.sort_values(by=['Country_Region','Province_State','Date'])
data=data.rename(columns={'Unnamed: 0':'Id'})
data = data.reset_index(drop=True)
data['Id'] = data.index + 1
c=data

dates=c['Date']

c=calculated_dr_features(c,dates)
c=calculate_diff_features(c,dates,'New Cases')
c=calculate_diff_features(c,dates,'New Deaths')
values={'Province/State': " "}
c.fillna(value=values,inplace=True)

for k in range (len(c)):
    if c['New Cases'][k]<0:
        c['New Cases'][k]= c['ConfirmedCases'][k]
for k in range (len(c)):
    if c['New Deaths'][k]<0:
        c['New Deaths'][k]=c['Fatalities'][k]
    
c=calculate_pct_features(c,dates)

c['BreakOut']=0
for p in range(len(c)):
    if c['Date'][p]== '2020-01-22' and c['ConfirmedCases'][p]!=0:
        c['BreakOut'][p]=1
    elif c['ConfirmedCases'][p]>0:
        c['BreakOut'][p]= c['BreakOut'][p-1]+1
 
c=c.rename(columns={'Province_State':'Province/State','Country_Region':'Country/Region','ConfirmedCases':'Confirmed','Fatalities':'Deaths'})  

#***********************************************************************************

#fix the datashet and select your feature

#get the data and store them to a data frame
data=c

#fix the sort format of the data to match kaggle's dataset
data=data.sort_values(by=['Country/Region','Province/State','Date'])
data = data.reset_index(drop=True)
data['Id'] = data.index + 1


#convert text data to categorical
data=convert_to_categorical(data,'Province/State')
data=convert_to_categorical(data,'Country/Region')
data=convert_to_categorical(data,'Date')


#Create the final dataFrame with our features
W=data[['Province/State','Country/Region','Date','Confirmed','Deaths','Death Rate','New Confirmed pct','New Deaths pct','BreakOut']]


#**********************************************************************************

#select your desire window!!! 
t_window = 8
#*************************

#get the number of diff country/region
country=0
for i in range(len(c)):
    if c["Date"][i]=="2020-01-22":
        country+=1
        
#**************************************

#Get the Labels for train

#shift the labels with the window 
max_date =W['Date'].max()+1
labels=data[['Date','Confirmed','Deaths']]
old = 0
new = max_date

for i in range(1,country+1):
    new=max_date*i
    labels[old:new] = labels[old:new].shift(-t_window,fill_value=0)
    old=new
    
#delete last day that we dont know the label 
Y=labels.loc[labels['Date'] != 0] 
Y = Y.reset_index(drop=True)
Y_C=Y[['Confirmed']]
Y_D=Y[['Deaths']]
Y_C = np.array(Y_C)
Y_D = np.array(Y_D)


#Y now contains the labels for train

#********************************************************************************

#Seperate our Data in lists per Country
# T contains lists with the data of each country
W_C=data[['Province/State','Country/Region','Date','Confirmed','New Confirmed pct','BreakOut']]
W_D=data[['Province/State','Country/Region','Date','Deaths','Death Rate','New Deaths pct','BreakOut']] #'New Cases','New Deaths'
W_C = np.array(W_C)
W_D = np.array(W_D)
T_C = []
T_D=[]
old = 0
new = 0

for i in range(1,country+1):
    new = max_date*i
    T_C[old:new] = [W_C[old:new]]
    T_D[old:new] = [W_D[old:new]]
    old = new



#split our data in windows 
X_D = []
X_C = []
for i in range(len(T_C)):
    pi=0
    for k in range(max_date):
        if len(T_C[i][k:-1])<t_window:
            break
        else:
            X_C.append(T_C[i][k:k+t_window])
            X_D.append(T_D[i][k:k+t_window])
            pi=pi+1

X_C=np.array(X_C)                     
X_D=np.array(X_D)                     
#now X is a array of window separate by country
            
#*****************************************************************************************

#create a train test set of last 10 days of kaggle train in order to see how the model work

#change it if you want bigger test     
new_t = 10

#merge test train data
YC_train = []
YC_test = []
XC_train = []
XC_test = []
YD_train = []
YD_test = []
XD_train = []
XD_test = []
old = 0
new = max_date-t_window-10

for i in range(1,country+1):
    new=(max_date-t_window-10)*i
    XC_train[old:new]=X_C[old:new]
    XC_test[old:new_t] = X_C[new:(new+new_t)]
    YC_train[old:new] = Y_C[old:new]
    YC_test[old:new_t] = Y_C[new:(new+new_t)]
    XD_train[old:new]=X_D[old:new]
    XD_test[old:new_t] = X_D[new:(new+new_t)]
    YD_train[old:new] = Y_D[old:new]
    YD_test[old:new_t] = Y_D[new:(new+new_t)]
    old=new


# Convert lists to arrays too feed them to the NN
XC_train = np.asarray(XC_train)
XC_test = np.asarray(XC_test)
YC_train = np.asarray(YC_train)
YC_test = np.asarray(YC_test)
XD_train = np.asarray(XD_train)
XD_test = np.asarray(XD_test)
YD_train = np.asarray(YD_train)
YD_test = np.asarray(YD_test)

#**********************************************************************************

#model create 1

#CNN model 
model_C = Sequential()
model_C.add(Conv1D(32,3,strides=1,padding='same',activation='softplus',input_shape=(t_window,6)))
model_C.add(Conv1D(32,3,strides=1,padding='same',activation='softplus'))
model_C.add(Conv1D(16,3,strides=1,padding='same',activation='softplus'))
#model.add(BatchNormalization())

#model.add(Conv1D(32,3,strides=1,padding='same'))

#LSTM model
#model.add(Flatten())
model_C.add(LSTM(64, activation='softplus',kernel_regularizer=regularizers.l2(0.0007),recurrent_regularizer=regularizers.l2(0.04)))
#model_C.add(Dense(64,activation='softplus'))
model_C.add(Dense(32,activation='softplus'))
model_C.add(Dense(1,activation='linear'))
opt=optimizers.Adam(learning_rate=0.01)
model_C.compile(optimizer='adam', loss='msle')
model_C.summary()

#************************************************************************************

#compile and train the model
history_C = model_C.fit(X_C,Y_C, epochs=500, verbose=1 , batch_size = 76,shuffle=False)
plot_loss(history_C)

#************************************************************************************

#model evaluation

pred=np.round(model_C.predict(XC_test))
score=rmse(pred,YC_test)
print("C_RMSE : {}".format(score))

te = pred.clip(min=0)
te = abs(te)
act2 = np.sqrt(mean_squared_log_error( YC_test, te))
print("C_RMSE : {}".format(act2))

#*************************************************************************************
#**********************************************************************************

#model create 2

#CNN model 
model_D = Sequential()
model_D.add(Conv1D(16,3,strides=1,padding='same',activation='softplus',input_shape=(t_window,7)))
#model_D.add(Conv1D(16,3,strides=1,padding='same',activation='softplus'))
#model_D.add(Conv1D(16,3,strides=1,padding='same',activation='softplus'))
#model.add(BatchNormalization())

#model.add(Conv1D(32,3,strides=1,padding='same'))

#LSTM model
#model.add(Flatten())
model_D.add(LSTM(64, activation='softplus',kernel_regularizer=regularizers.l2(0.0008),recurrent_regularizer=regularizers.l2(0.001)))
#model_D.add(Dense(32,activation='softplus'))
#model_D.add(Dense(16,activation='softplus'))
model_D.add(Dense(1,activation='linear'))
opt=optimizers.Adam(learning_rate=0.01)
model_D.compile(optimizer='adam', loss='msle')
model_D.summary()

#************************************************************************************

#compile and train the model
history_D = model_D.fit(X_D,Y_D, epochs=500, verbose=1 , batch_size = 152,shuffle=False)
plot_loss(history_D)

#************************************************************************************
#model evaluation

pred=np.round(model_D.predict(XD_test))
score=rmse(pred,YD_test)
print("D_RMSE : {}".format(score))

te = pred.clip(min=0)
te = abs(te)
act2 = np.sqrt(mean_squared_log_error( YD_test, te))
print("D_RMSE : {}".format(act2))
#kaggle unpredicted data
#************************************************************************************


filepath="/kaggle/input/covid19-global-forecasting-week-4/test.csv"

k_test = pd.read_csv(filepath)
values={'Province/State': " "}
k_test.fillna(value=values,inplace=True)
k_test=convert_to_categorical(k_test,'Province_State')
k_test=convert_to_categorical(k_test,'Country_Region')
k_test=convert_to_categorical_dates(k_test,'Date')

kC_test=k_test
kD_test=k_test
kC_test['Confirmed']=''
kC_test['New Confirmed pct']=''
kC_test['BreakOut']=''
#k_test['New Cases']=''
#k_test['New Deaths']=''

kD_test['Deaths']=''
kD_test['Death Rate']=''
kD_test['New Deaths pct'] =''
kD_test['BreakOut']=''
#fill k_test with the overlap data 
pos_k = 0
overlap = 13
pos_d = max_date-overlap
for i in range(1,country+1):
    kC_test['Confirmed'][pos_k:pos_k+overlap] =  data['Confirmed'][pos_d:pos_d+overlap]
    kC_test['New Confirmed pct'][pos_k:pos_k+overlap] = data['New Confirmed pct'][pos_d:pos_d+overlap]
    kC_test['BreakOut'][pos_k:pos_k+overlap]= data['BreakOut'][pos_d:pos_d+overlap]
    #k_test['New Cases'][pos_k:pos_k+overlap] = data['New Cases'][pos_d:pos_d+overlap]
    #k_test['New Deaths'][pos_k:pos_k+overlap] = data['New Deaths'][pos_d:pos_d+overlap]
    kD_test['Deaths'][pos_k:pos_k+overlap] = data['Deaths'][pos_d:pos_d+overlap]
    kD_test['Death Rate'][pos_k:pos_k+overlap] = data['Death Rate'][pos_d:pos_d+overlap]
    kD_test['New Deaths pct'][pos_k:pos_k+overlap]  = data['New Deaths pct'][pos_d:pos_d+overlap]
    kD_test['BreakOut'][pos_k:pos_k+overlap]= data['BreakOut'][pos_d:pos_d+overlap]
    
  
    pos_k=pos_k + 43
    pos_d = pos_d + max_date
  

   
kC_test = kC_test[['Province_State','Country_Region','Date','Confirmed','New Confirmed pct','BreakOut']]    
kD_test = kD_test[['Province_State','Country_Region','Date','Deaths','Death Rate','New Deaths pct','BreakOut']]   

kC_test = np.asarray(kC_test)
kD_test = np.asarray(kD_test)

#now k_test have the overlap days fiiled for all country and we can predict the others

#*******************************************************************************************************

#Predict rest test days
inp=[]
for i in range(0,country):
    for k in range(overlap-t_window,43-t_window):
        z=i*43+k
        inpc = [kC_test[z:z+t_window]]
        tmpc = model_C.predict(np.array(inpc))
        tmpc = np.round(tmpc.clip(min=0))
        kC_test[z+t_window,3] = tmpc
        kC_test[z+t_window,4] = weird_division(kC_test[z+t_window,3],kC_test[z+t_window-1,3]) #new pct confirmed
       
    
        #k_test[z+t_window,6] = k_test[z+t_window,3] - k_test[z+t_window-1,3] #new case confirmed
        #k_test[z+t_window,7] = k_test[z+t_window,4] - k_test[z+t_window-1,4] #new case dead
        
        inpd = [kD_test[z:z+t_window]]
        tmpd = model_D.predict(np.array(inpd))
        tmpd = np.round(tmpd.clip(min=0))
        kD_test[z+t_window,3] = tmpd
        kD_test[z+t_window,4] = weird_division(kD_test[z+t_window,3],kC_test[z+t_window,3]) #death rate
        kD_test[z+t_window,5] = weird_division(kD_test[z+t_window,3],kD_test[z+t_window-1,3]) #new pct dead
        #k_test[z+t_window,6] = k_test[z+t_window,3] - k_test[z+t_window-1,3] #new case confirmed
        #k_test[z+t_window,7] = k_test[z+t_window,4] - k_test[z+t_window-1,4] #new case dead
        
        
        
        if kC_test[z+t_window-1,-1] ==0 and kC_test[z+t_window,3]!=0:
                kC_test[z+t_window,-1] = 1
                kD_test[z+t_window,-1] = 1
        else:
                kC_test[z+t_window,-1] = kC_test[z+t_window-1,-1] +1
                kD_test[z+t_window,-1] = kD_test[z+t_window-1,-1] +1
        kC_test[z+t_window] =  kC_test[z+t_window].clip(min=0)
        kD_test[z+t_window] =  kD_test[z+t_window].clip(min=0)
        

#*****************************************************************************************
        
#Create the submission file kaggle wants

filepath="/kaggle/input/covid19-global-forecasting-week-4/submission.csv"


sub = pd.read_csv(filepath)
sub['ConfirmedCases'] = kC_test[:,3].astype(int)
sub['Fatalities'] = kD_test[:,3].astype(int)
sub.to_csv("submission.csv",index = False)

#submission ready for commite

#*****************************************************************************************
