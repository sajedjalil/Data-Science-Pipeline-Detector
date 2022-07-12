# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import sklearn 
import datetime
import matplotlib.pyplot as plt
from sklearn import model_selection
from numpy import array


# In[2]:


crime_test = pd.read_csv("../input/test.csv", sep=',',error_bad_lines=False)
crime_train = pd.read_csv("../input/train.csv", sep=',',error_bad_lines=False)


# ## Address preprocessing

# In[3]:


dict = {}
for index, row in crime_train.iterrows():
    obj = row["Address"].replace(" ","")
    in0 = obj.find('/')
    len0 = len(obj)
    ele = []
    ele.append(obj[len0-2: len0])
    if in0!=-1:
        ele.append(obj[in0-2:in0])
    for ss in ele:
        if dict.__contains__(ss):
            dict[ss] = dict[ss] + 1
        else:
            dict[ss] = 1


# In[4]:


def getClass0(number):
    if number < 1000:
        return 0
    if number < 10000:
        return 1
    return 2
def extractRegion(row):
    ss = row.replace(" ","")
    index = ss.find('/')
    len0 = len(ss)
    str1 = dict[ss[len0-2: len0]]
    if index != -1:
        str0 = dict[ss[index-2:index]]
        if str0 > str1:
            t = str0
            str0 = str1
            str1 = t
        return str(getClass0(str0)) + "/" + str(getClass0(str1))
    else:
        return "Block "+ str(getClass0(str1))


# In[5]:


crime_train['Region'] = crime_train.apply(lambda row: extractRegion(row['Address']), axis=1)
crime_test['Region'] = crime_test.apply(lambda row: extractRegion(row['Address']), axis=1)


# ## Date preprocessing(including Hour and Month)

# In[6]:


def getHourClass(hour):
    return "H"+str(hour)
    
# Extract hour from date colum
def extractHours(row):
    date, time = row.split()
    h = int(time[0:2])
    return getHourClass(h)

crime_train['Hour'] = crime_train.apply(lambda row: extractHours(row['Dates']), axis=1)
crime_test['Hour'] = crime_test.apply(lambda row: extractHours(row['Dates']), axis=1)


# In[7]:


def getMonthClass(month):
    return 'M'+str(month)
    
# Extract hour from date colum
def extractMonths(row):
    date, time = row.split()
    h = int(date[5:7])
    return getMonthClass(h)

crime_train['Month'] = crime_train.apply(lambda row: extractMonths(row['Dates']), axis=1)
crime_test['Month'] = crime_test.apply(lambda row: extractMonths(row['Dates']), axis=1)


# ## generate the vectors for training set by dummy

# In[9]:


from sklearn import preprocessing

#用LabelEncoder对不同的犯罪类型编号
leCrime = preprocessing.LabelEncoder()
crime = leCrime.fit_transform(crime_train.Category)

#因子化星期几，街区，小时等特征
days = pd.get_dummies(crime_train.DayOfWeek)
district = pd.get_dummies(crime_train.PdDistrict)
hour = pd.get_dummies(crime_train.Hour) 
month = pd.get_dummies(crime_train.Month)
region = pd.get_dummies(crime_train.Region)
#组合特征
trainData = pd.concat([hour, days, district, region, month], axis=1)
trainData['crime']=crime


# ## generate the vectors for test set by dummy

# In[11]:


#对于测试数据做同样的处理
days = pd.get_dummies(crime_test.DayOfWeek)
district = pd.get_dummies(crime_test.PdDistrict)

hour = pd.get_dummies(crime_test.Hour) 
region = pd.get_dummies(crime_test.Region)
month = pd.get_dummies(crime_test.Month)

testData = pd.concat([crime_test.Id, hour, days, district, region, month], axis=1)


# ## Random Forest model

# In[15]:


features = ['H0', 'H1', 'H2', 'H3', 'H4','H5','H6','H7','H8','H9','H10','H11','H12','H13','H14','H15','H16','H17',\
                    'H18','H19','H20','H21','H22','H23','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','BAYVIEW','CENTRAL','INGLESIDE','MISSION',\
           'NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','0/0','0/1','0/2','1/1','1/2','2/2','Block 0','Block 1','Block 2',\
           'M1', 'M2', 'M3', 'M4','M5','M6','M7','M8','M9','M10','M11','M12']
from sklearn.ensemble import RandomForestClassifier

# Parameters to be tested
parameters = {'n_estimators':100, 
              'criterion':'gini', 
              'max_depth':15}

# Comparison
rf = RandomForestClassifier()
rf.set_params(**parameters)

f = rf.fit(trainData[features], trainData['crime'])
predicted = np.array(f.predict_proba(testData[features]))


# ## generate the result of predict

# In[16]:


#np.round_(predicted, decimals=2, out=predicted)
colmn = ["ARSON","ASSAULT","BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT","DRIVING UNDER THE INFLUENCE","DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING","FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON","NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE","ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE","STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT","WARRANTS","WEAPON LAWS"]
result = pd.DataFrame(predicted, columns=colmn)

result.to_csv(path_or_buf="resultbaye.csv",index=True, index_label = 'Id')