# %% [code] {"id":"4pdFe6M3mEvd"}
#importing the important libraries
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error ,mean_absolute_error
import datetime
import operator
plt.style.use('seaborn')
%matplotlib inline

# %% [code] {"id":"5H3GNngVnKik","outputId":"e09d41bd-62cb-4d9f-9b32-9af06ae12f03"}
from google.colab import files
uploaded = files.upload()
confirmed_cases = pd.read_csv('data.csv')
confirmed_cases.head(477)

# %% [code] {"id":"OyMmBhdqpgPO","outputId":"6de56c0c-2bbd-4c36-e4f7-8ae9d4afe3f2"}
from google.colab import files
uploaded = files.upload()
deaths_reported = pd.read_csv('death.csv')
deaths_reported.head()

# %% [code] {"id":"jDzs3prrqfM3","outputId":"2ed1bce0-2835-4e6e-f716-145197dcf02a"}
from google.colab import files
uploaded = files.upload()
recovered_cases = pd.read_csv('recovered.csv')
recovered_cases.head()

# %% [code] {"id":"Esmxs1mIrFvS","outputId":"14fe156d-16b3-4348-a6f6-17693efa9517"}
#diplay the headset of the dataset
confirmed_cases.head()

# %% [code] {"id":"lXq61mX8rRzp","outputId":"00240367-f91e-4a7c-86e8-d6210e2fb0c0"}
deaths_reported.head()

# %% [code] {"id":"Lks1cMZRreDA","outputId":"6f18fd3d-0db2-4afc-8108-88686f73ef49"}
recovered_cases.head()

# %% [code] {"id":"N_Kr_7W-ri5B","outputId":"0dbcbbdc-3559-496c-a6bb-a2dcc9e6667e"}
#extracting all the columns using the .keys  function
cols =confirmed_cases.keys()
cols

# %% [code] {"id":"9OLdepFF3skI"}


# %% [code] {"id":"XRoJNj0Js_-p"}
#extracting only the dates columns that have the information of confirmed,deaths  and recovered cases
confirmed = confirmed_cases.loc[:, cols[4]:cols[-1]]
# we need all the  parameters from the 4th column to the -1 column

# %% [code] {"id":"y5w97jnMul2v"}
deaths = deaths_reported.loc[:, cols[4]:cols[-1]]

# %% [code] {"id":"qLei1iLbu2lt"}
recoveries = recovered_cases.loc[:, cols[4]:cols[-1]]

# %% [code] {"id":"tRCQfkQbvDD2","outputId":"be878fea-f620-4662-ba38-f5c680292ef1"}
# check the head of the outbreak cases
confirmed.head()

# %% [code] {"id":"sY9RjxIav2RY"}
#finding the total confirmed cases,death caess and the recoverd cases and append them on the 4th empty lists
#Also,calculate the toatal mortality rate which is death_sum/confirmed_cases
dates = confirmed.keys()
world_cases = []
total_deaths = []
mortality_rate = []
total_recovered = []

for  i in dates:
  confirmed_sum= confirmed[i].sum()
  death_sum =deaths[i].sum()
  recovered_sum =recoveries[i].sum()
  world_cases.append(confirmed_sum)
  total_deaths.append(death_sum)
  mortality_rate.append(death_sum/confirmed_sum)
  total_recovered.append(recovered_sum)

# %% [code] {"id":"GgySskwHyJ7-","outputId":"8a0b8f71-5a5b-4ef5-de6d-844ab79ceac4"}
#lets display the each of the new created varaiable
confirmed_sum

# %% [code] {"id":"8OoplkOyyXu_","outputId":"dd9abdaa-27eb-47aa-c1df-86376ee6498e"}
death_sum

# %% [code] {"id":"nCsWcS0Jygkj","outputId":"2fb05a79-1a73-4470-a8af-88d96459ec48"}
recovered_sum

# %% [code] {"id":"xJ_vuoqMynl5","outputId":"aaca6e35-1c24-4298-f12a-f43f0dcdca62"}
world_cases

# %% [code] {"id":"MHRh1vNizImG"}
#convert all the dates and the cases in the formm of the numpy arrays
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases = np.array(world_cases).reshape(-1,1)
total_deaths = np.array(total_deaths).reshape(-1,1)
total_recoverd = np.array(total_recovered).reshape(-1,1)

# %% [code] {"id":"F1mO9eFF0UNv","outputId":"ae467511-2109-4304-87ab-1c879ba7e393"}
days_since_1_22

# %% [code] {"id":"i6Dfg3KH0aP1","outputId":"bf479acc-0578-4c0d-c910-ab194bfa1ebe"}
world_cases

# %% [code] {"id":"K4bBR6vK0abt","outputId":"0310bf7b-a2d2-4cf0-ce4f-bf656d4477bd"}
total_deaths

# %% [code] {"id":"x4hiS5Wk0anw","outputId":"fb26ea61-a4f7-4f04-ca7e-ff806026ad6f"}
total_recoverd

# %% [code] {"id":"2n0u06FHyZeM"}
# future forecasting for the next  10  days
days_in_future = 10
future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast[:-10]

# %% [code] {"id":"EwsESJxQ1yC3","outputId":"de7c682a-4799-46fd-cd95-d29ce4b88aca"}
future_forecast

# %% [code] {"id":"J-GOI3Z82a8z"}
#convert  all the integers  into datetime for visualisation
start = '1/12/2020'
start_date = datetime.datetime.strptime(start,'%m/%d/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
  future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

# %% [code] {"id":"FWRxatLT5Vax"}
#for visualisation with the latest date of 15th of march
latest_confirmed = confirmed_cases[dates[-1]]
latest_deaths = deaths_reported[dates[-1]]
latest_recoveries = recovered_cases[dates[-1]]

# %% [code] {"id":"oK3MkMqG6OGC","outputId":"886ceac7-b8f8-4740-af14-6a239752585d"}
latest_confirmed

# %% [code] {"id":"t5KUTIxA6v6Q","outputId":"4f94117a-9f2b-473a-f51a-b71eb02f9f26"}
latest_deaths

# %% [code] {"id":"KpCx2tHJ60Hp","outputId":"ef6f4d7d-ffa7-4e09-dc6f-9bddc6b02570"}
latest_recoveries

# %% [code] {"id":"Rs_exVMB64ka","outputId":"5bf152fa-8726-4821-e5ff-657cedaf7b55"}
#finding the list of unique countries
unique_countries = list(confirmed_cases['Country/Region'].unique())
unique_countries

# %% [code] {"id":"Tg3GPo-17XmW"}
#the next  line of code will basically calculate the total number pf confirmed cases by each country
country_confirmed_cases = []
no_cases = []
for  i  in unique_countries:
  cases = latest_confirmed[confirmed_cases['Country/Region']==i].sum()
  if cases > 0:
    country_confirmed_cases.append(cases)
  else:
      no_cases.append(i)

for i in no_cases:
  unique_countries.remove(i)
unique_countries = [k for k,v in sorted(zip(unique_countries,country_confirmed_cases)]
for i in range(len(unique_countries)):
  country_confirmed_cases[i] = latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()  

    

# %% [code] {"id":"ZQHiMd5vwpGM"}
#number of cases per country/region
#finding the list of unique countries
unique_countries = list(confirmed_cases['Country/Region'].unique())
print('Confirmed cases by countries/regions:')
for i in range(len(unique_countries)):
  print(f'{unique_countries[i]}:{confirmed_cases[i]}cases')

# %% [code] {"id":"7tOQLd0v9qF3"}
#visulise the count
import seaborn as sns
sns.countplot(confirmed_cases['	Country/Region'],label='count')

# %% [code] {"id":"nizG9GBdqzR1"}


# %% [code] {"id":"OwL3EfJwBLnr","outputId":"ee749d48-312e-422e-b9de-9479cdbabe47"}
#visualise the corerelation
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(confirmed_cases.iloc[:,1:12].corr(),annot=True, fmt='.0%')


# %% [code] {"id":"bELTlR4btEjv","outputId":"91567d8d-3415-4811-8ea2-9c2b1cccab87"}
#visualise the corerelation
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(deaths_reported.iloc[:,1:12].corr(),annot=True, fmt='.0%')


# %% [code] {"id":"HsuJCS7jtAbt"}


# %% [code] {"id":"9982KwIpttWN","outputId":"9312dc4b-739c-43b3-a57f-5570f04d08d3"}
#visualise the corerelation
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(recovered_cases.iloc[:,1:12].corr(),annot=True, fmt='.0%')


# %% [code] {"id":"0YxUbSnQt4t3"}


# %% [code] {"id":"v-h2Lua85T0P","outputId":"39cd262f-4280-4796-f6f2-ceddd712cdc2"}
# create a pair plot
sns.pairplot(confirmed_cases.iloc[:,1:6])

# %% [code] {"id":"TxPmIvviuMA7"}


# %% [code] {"id":"o90sCWAOuRLL","outputId":"8b2a8de0-f155-43e8-aa29-6ce9272cac9d"}
# create a pair plot
sns.pairplot(deaths_reported.iloc[:,1:6])

# %% [code] {"id":"2RF3kJuHumCj"}


# %% [code] {"id":"7FfDCxAT0MLF","outputId":"7d810d06-6bb9-40f4-ecce-cfc5ce243852"}
#get the corelation of the coluns
confirmed_cases.iloc[:,1:12].corr()

# %% [code] {"id":"VgX5cVoKCDiR"}
# split the dataset intpo independent and dependent(Y) data sets
X = confirmed_cases.iloc[:,2:31].values
Y = confirmed_cases.iloc[:,1].values

# %% [code] {"id":"AJ8OoEgrtqW7"}


# %% [code] {"id":"sFLOQknjERGM"}
# Split the datset into 75 percent raining and 25 percent testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state = 0)

# %% [code] {"id":"dftH-wsKwQ_k"}


# %% [code] {"id":"kH9R3SlKFkqE","outputId":"264e7d6a-2771-4d3e-8a40-b82d4d03ba59"}
# scale the data(feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)
X_train

# %% [code] {"id":"rZHAJmmVwVvJ"}


# %% [code] {"id":"gPCdx8T1G9nn"}
# create a function for the model
def models(X_train,Y_train):
  #Logistic regression
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state=0)
  log.fit(X_train,Y_train)

  #Decisssion tree
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion ='entropy', random_state=0)
  tree.fit(X_train,Y_train)

  #Random forest classifire
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10,criterion ='entropy',random_state=0)
  forest.fit(X_train, Y_train)

  #print the model accuracy on the training data
  print('[0]Logistic Regression Training Accuracy:',log.score(X_train,Y_train))
  print('[1]Decision tree classifier training Accuracy:',tree.score(X_train,Y_train))
  print('[2]Random forest classifier Training Accuracy:',forest.score(X_train,Y_train))

  return log,tree,forest



# %% [code] {"id":"wtb1i0FbwrQb"}


# %% [code] {"id":"-Iln8wZDL9UR","outputId":"e773b4fe-1d12-496d-8bdf-3656dbf1cba0"}
#getting all of the models
model = models(X_train, Y_train)


# %% [code] {"id":"4tKsnpYOw08b"}


# %% [code] {"id":"MtkAJuZLMlNR","outputId":"0b35d732-4489-4240-9516-bb803a54e86e"}
#test model accuracy on test dat on confusion matrix
from sklearn.metrics import confusion_matrix

for i in range( len(model) ):
  print('Model',i)
  cm = confusion_matrix(Y_test,model[1].predict(X_test))

  TP = cm[0][0]
  TN = cm[1][1]
  FN = cm[1][0]
  FP =  cm[0][1]
  print(cm)
  print('Testing Accuracy',(TP + TN)/(TP + TN + FN + FP))

# %% [code] {"id":"8bmXwdyqxGus"}


# %% [code] {"id":"n-hnDtTKSjK0","outputId":"229deb2f-d7b6-4ba3-dcb3-2d7a2fb0c5ff"}
# show another way to get the atrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range( len(model) ):
  print('Model',i)
  print( classification_report(Y_test,model[i].predict(X_test)))
  print( accuracy_score(Y_test,model[i].predict(X_test)))

# %% [code] {"id":"WYkBaxmsxPa7"}


# %% [code] {"id":"KTK1jX_VVkZj","outputId":"f15d2215-b542-4466-b117-01b47e3d6160"}
# print the prediction of Random forest classifiesr
pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)