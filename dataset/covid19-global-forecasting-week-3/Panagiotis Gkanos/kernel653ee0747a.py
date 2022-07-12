import warnings
warnings.filterwarnings("ignore")

import random
random.seed(42)
from numpy.random import seed
seed(1)
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
from keras import regularizers

def weird_division(a,b):
    if b == 0:
        return 0
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





filepath='/kaggle/input/covid19-global-forecasting-week-3/train.csv'

data=pd.read_csv(filepath)
data=data.sort_values(by=['Country_Region','Province_State','Date'])
data=data.rename(columns={'Unnamed: 0':'Id'})
data = data.reset_index(drop=True)#, inplace=True)
data['Id'] = data.index + 1
c=data
'''
for k in range (len(c)):
    if ((k%77)!=0):
        if c['ConfirmedCases'][k]-c['ConfirmedCases'][k-1]<0:
            c['ConfirmedCases'][k]= c['ConfirmedCases'][k-1]
        if c['Fatalities'][k]-c['Fatalities'][k-1]<0:
            c['Fatalities'][k]= c['Fatalities'][k-1]
'''
dates=c['Date']

c=calculated_dr_features(c,dates)
c=calculate_diff_features(c,dates,'New Cases')
c=calculate_diff_features(c,dates,'New Deaths')

values={'Province/State': "None"}
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
count=0
for i in range(len(c)):
    if c["Date"][i]=="2020-01-22":
        count+=1
print(count)   
        
        
        
        
        
        

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
values={'Province/State': "None"}
data.fillna(value=values,inplace=True)

#fix the sort format of the data to match kaggle's dataset
data=data.sort_values(by=['Country/Region','Province/State','Date'])
data = data.reset_index(drop=True)#, inplace=True)
data['Id'] = data.index + 1


#convert text data to categorical
data=convert_to_categorical(data,'Province/State')
data=convert_to_categorical(data,'Country/Region')
data=convert_to_categorical(data,'Date')


#Create the final dataFrame with our features
W=data[['Province/State','Country/Region','Date','Confirmed','New Cases','New Confirmed pct','BreakOut']]


#get the labels
#shift the labels with the window 
max_date =W['Date'].max()+1
labels=data[['Date','Confirmed']]
old = 0
new = max_date
t_window = 3
for i in range(1,307):#CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    new=max_date*i
    labels[old:new] = labels[old:new].shift(-t_window,fill_value=0)
    old=new

Y=labels.loc[labels['Date'] != 0] #delete last day that we dont know the label 
Y = Y.reset_index(drop=True)
Y=Y[['Confirmed']]
Y = np.array(Y)


#Seperate our Data in lists per Country
# T contains lists with the data of each country
W = np.array(W)
T = []
old = 0
new = 0

for i in range(1,307):#CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                       

#**********************************************************
            
#split data

#merge test train data
Y_train = []
Y_test = []
X_train = []
X_test = []

old = 0
new = max_date-t_window-10
new_t = 10
for i in range(1,307): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

X = np.asarray(X)

#************************************************************

#model create

    
#LSTM model  for Confirm Cases    
model = Sequential()
model.add(LSTM(64, input_shape=(3,7),recurrent_initializer = 'orthogonal',kernel_regularizer=regularizers.l2(0.001), activation='softplus'))
model.add(Dense(32, activation='softplus'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
model.summary()
# Train model
history = model.fit(X,Y, epochs=800, verbose=1 , batch_size = 140,shuffle=False)
plot_loss(history)
 
    
#LSTM model for Fatalities     
model_F = Sequential()
model_F.add(LSTM(64, activation='softplus', input_shape=(3,7),recurrent_initializer = 'orthogonal'))
model_F.add(Dense(1))
model_F.compile(optimizer='adam', loss='msle')

# Train model
history_F = model_F.fit(X,Y, epochs=500, verbose=1 , batch_size = 140,shuffle=False)
plot_loss(history_F)

#*****************************************************
#Function to calulate logarithmic sq error         
def rmse(pred,val):
  x = pred.clip(min=0)
  x = abs(x)
  logpred=np.log(x+1)
  logval=np.log(val+1)
  rmse=np.sqrt(np.sum((logpred-logval)**2)/val.shape[0])
  return rmse

#model evaluation
pred=np.round(model.predict(X_test))
score=rmse(pred,Y_test)
print("RMSE : {}".format(score))

act1 = np.sqrt(mean_squared_log_error( Y_test,abs(pred)))

te = pred.clip(min=0)
te = abs(te)
act2 = np.sqrt(mean_squared_log_error( Y_test, te))
'''
#dif model 
model_cases = Sequential()
model_cases.add(LSTM(60, input_shape=(3,7), activation='softplus'))
model_cases.add(Dense(1, activation='sigmoid'))
model_cases.compile(loss='mse', optimizer='adam')

history2 = model_cases.fit(X_train, Y_train, batch_size=67, epochs=500)
'''
#*******************************************************

#kaggle unpredicted data for Confirm

filepath='/kaggle/input/covid19-global-forecasting-week-3/test.csv'

k_test = pd.read_csv(filepath)

k_test['Confirmed']=''
k_test['New Cases']=''
k_test['New Confirmed pct']=''
k_test['BreakOut']=''


pos_k = 0
overlap = 13
pos_d = max_date-overlap
for i in range(1,307):
    k_test['Confirmed'][pos_k:pos_k+overlap] =  data['Confirmed'][pos_d:pos_d+overlap]
    k_test['New Cases'][pos_k:pos_k+overlap] = data['New Cases'][pos_d:pos_d+overlap]
    k_test['New Confirmed pct'][pos_k:pos_k+overlap]=data['New Confirmed pct'][pos_d:pos_d+overlap]
    k_test['BreakOut'][pos_k:pos_k+overlap]= data['BreakOut'][pos_d:pos_d+overlap]
    
    pos_k=pos_k + 43
    pos_d = pos_d + max_date
  

   
k_test = k_test[['Province_State','Country_Region','Date','Confirmed','New Cases','New Confirmed pct','BreakOut']]    

values = {'Province_State' : "None"}
k_test.fillna(value=values,inplace=True)

k_test=convert_to_categorical(k_test,'Province_State')
k_test=convert_to_categorical(k_test,'Country_Region')
k_test=convert_to_categorical_dates(k_test,'Date')

k_test = np.asarray(k_test)


inp=[]


for i in range(0,306):
    for k in range(overlap-3,40):
        z=i*43+k
        inp = [k_test[z:z+3]]
        tmp = model.predict(np.array(inp))
        tmp = np.round(tmp.clip(min=0))
        k_test[z+3,3] = tmp[0,0]
        k_test[z+3,4] = k_test[z+3,3] - k_test[z+2,3] #new case confirmed
        k_test[z+3,5] = weird_division(k_test[z+3,3],k_test[z+2,3])#new conf pct
        
        if k_test[z+2,-1] ==0 and k_test[z+3,3]!=0:
                k_test[z+3,-1] = 1
        else:
                k_test[z+3,-1] =k_test[z+2,-1] + 1
        k_test[z+3] =  k_test[z+3].clip(min=0)
        

#kaggle unpredicted data for Fatalities

filepath='/kaggle/input/covid19-global-forecasting-week-3/test.csv'

k_test_F = pd.read_csv(filepath)


k_test_F['Deaths']=''
k_test_F['New Deaths']=''
k_test_F['New Deaths pct'] =''
k_test_F['BreakOut']=''

pos_k = 0
overlap = 13
pos_d = max_date-overlap
for i in range(1,307):
    
    k_test_F['Deaths'][pos_k:pos_k+overlap] = data['Deaths'][pos_d:pos_d+overlap]
    k_test_F['New Deaths'][pos_k:pos_k+overlap] = data['New Deaths'][pos_d:pos_d+overlap]
    k_test_F['New Deaths pct'][pos_k:pos_k+overlap]  = data['New Deaths pct'][pos_d:pos_d+overlap]
    k_test_F['BreakOut'][pos_k:pos_k+overlap]= data['BreakOut'][pos_d:pos_d+overlap]
  
    pos_k=pos_k + 43
    pos_d = pos_d + max_date
  

   
k_test_F = k_test_F[['Province_State','Country_Region','Date','Deaths','New Deaths','New Deaths pct','BreakOut']]    
values = {'Province_State' : "None"}
k_test_F.fillna(value=values,inplace=True)
k_test_F=convert_to_categorical(k_test_F,'Province_State')
k_test_F=convert_to_categorical(k_test_F,'Country_Region')
k_test_F=convert_to_categorical_dates(k_test_F,'Date')

k_test_F = np.asarray(k_test_F)

t= 42
inp=[]


for i in range(0,306):
    for k in range(overlap-3,40):
        z=i*43+k
        inp = [k_test_F[z:z+3]]
        tmp = model_F.predict(np.array(inp))
        tmp = np.round(tmp.clip(min=0))
        k_test_F[z+3,3] = tmp[0,0]
        k_test_F[z+3,4] = k_test_F[z+3,3] - k_test_F[z+2,3] #new case dead
        k_test_F[z+3,5] = weird_division(k_test_F[z+3,3],k_test_F[z+2,3]) #new pct dead
        
        if k_test_F[z+2,-1] ==0 and k_test_F[z+3,3]!=0:
                k_test_F[z+3,-1] = 1
        else:
                k_test_F[z+3,-1] = k_test_F[z+2,-1] +1
        k_test_F[z+3] =  k_test_F[z+3].clip(min=0)
        


#make sub file

filepath='/kaggle/input/covid19-global-forecasting-week-3/submission.csv'


sub = pd.read_csv(filepath)
sub['ConfirmedCases'] = k_test[:,3].astype(int)
sub['Fatalities'] = k_test_F[:,3].astype(int)
sub.to_csv('submission.csv',index = False)

filepath='/kaggle/input/covid19-global-forecasting-week-3/test.csv'
test=pd.read_csv(filepath)
test['ConfirmedCases'] = sub['ConfirmedCases']



#plots
#ind = test.loc[test['Province_State']=='Hubei']
ind = test.loc[test['Country_Region']=='Germany']
m = ind['ForecastId'].max()
mi = ind['ForecastId'].min()
confCases = test[mi:m]['ConfirmedCases']
plt.plot(confCases, label = 'Confirmed Cases')
plt.xlabel('Predictions Dates since 19/03')
plt.ylabel('Confirmed Cases')
plt.title('Confirmed Cases')
