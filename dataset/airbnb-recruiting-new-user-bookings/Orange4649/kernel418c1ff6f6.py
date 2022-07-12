# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
import math
import seaborn as sb
import matplotlib.pyplot as plt
import os

customer = pd.read_csv('../input/age_gender_bkts.csv')  # 讀取訓練數據
country = pd.read_csv('../input/countries.csv')  # 讀取訓練數據
test_users = pd.read_csv('../input/test_users.csv')  # 讀取訓練數據
train_users = pd.read_csv('../input/train_users_2.csv')  # 讀取訓練數據
def dayd(a,b):
    return (pd.Timestamp(a)-pd.Timestamp(b)).days

"""
資料前處理
"""
def data_encoder(data):
    #處理gender
    data = data.replace('-unknown-',0)
    data = data.replace('OTHER',0)
    data = data.replace('MALE',1)
    data = data.replace('FEMALE',2)
    #處理極端值age
    data['age'] = data['age'].astype(float)
    mean_age=int(data.age.mean())
    data.age = data.age.map(lambda x:extreme_value(x,mean_age))
    #處理season
    data.date_account_created = data.date_account_created.map(lambda x:season(x))
    data.rename(columns={'date_account_created':'season'})
    return data
def extreme_value(x,mean_age):
    if(math.isnan(x)):
        return mean_age
    elif(x>100):
        return 100
    elif(x<15):
        return 15
    else:
        return x
def season(x):
    if(int(x[5:7])<=3):#冬
        return 1
    if(int(x[5:7])<=6):#春
        return 2
    if(int(x[5:7])<=9):#夏
        return 3
    if(int(x[5:7])<=12):#秋
        return 4   
def label_encoder(data):
    #處理label分類
    """
    AU:澳洲,CA:加拿大,DE:德國,ES:西班牙,FR:法國
    GB:英國,IT:義大利,NDF:不知道,NL:荷蘭,PT:葡萄牙,US:美國
    """
    data = data.replace('US','0')
    data = data.replace('CA','1')
    data = data.replace('DE','2')
    data = data.replace('ES','3')
    data = data.replace('FR','4')
    data = data.replace('GB','5')
    data = data.replace('IT','6')
    data = data.replace('NL','7')
    data = data.replace('PT','8')
    data = data.replace('AU','9')
    data = data.replace('other','10')
    data = data.replace('NDF','11')
    return data

def label_encoder2(data):
    #處理label分類
    """
    AU:澳洲,CA:加拿大,DE:德國,ES:西班牙,FR:法國
    GB:英國,IT:義大利,NL:荷蘭,PT:葡萄牙
    """
    data = data.replace('CA','0')
    data = data.replace('DE','1')
    data = data.replace('ES','2')
    data = data.replace('FR','3')
    data = data.replace('GB','4')
    data = data.replace('IT','5')
    data = data.replace('NL','6')
    data = data.replace('PT','7')
    data = data.replace('AU','8')
    return data   

"""
1-1 用customer統計一些旅遊人數
"""
customer['population_in_thousands']=customer['population_in_thousands'].astype(float)
country_population=customer[['country_destination','population_in_thousands']]
country_population=country_population.groupby('country_destination').sum()
country_population=country_population.sort_values(by='population_in_thousands',ascending=False)
print(country_population)
"""
1-2 從註冊帳戶分析
"""
train_users_2=train_users.drop(columns=['country_destination'])
users = pd.concat([train_users_2,test_users], axis=0)
nan_count=users['date_first_booking'].value_counts(dropna=False)[0] #找出有多少nan值
used_rate=nan_count/(users['date_account_created'].count())*100
print("註冊完會訂房的比率 = "+str(used_rate)[0:5]+"%")
create_to_book = train_users[['date_account_created','date_first_booking']]
create_to_book = create_to_book.dropna(subset=['date_first_booking']) #把nan值去掉
create_to_book['span'] = create_to_book.apply(lambda row: dayd(row['date_first_booking'], row['date_account_created']), axis=1)
#dd有負值 代表先訂房才辦帳號
create_to_book=create_to_book[create_to_book["span"]>=0]
span_mean = create_to_book["span"].mean()
print("註冊完到訂房時間平均天數 = "+str(span_mean)[0:5]+"天")
users['signup_flow'] = users['signup_flow'].astype(float)
page_mean = users["signup_flow"].mean()
print("平均在第幾頁找到想要的房間 = "+str(page_mean)[0:4]+"頁")
filter_app = users['signup_app'] == 'Web' 
users_web = users[filter_app]
page_mean = users_web["signup_flow"].mean()
print("使用Web平均在第幾頁找到想要的房間 = "+str(page_mean)[0:4]+"頁")
#users.gender.unique()
#可以看有哪些值
users['age'] = users['age'].astype(float)
users['age'].describe()
#2014跟1應該是不可能
users_age = users.dropna(subset=['age']) #把nan值去掉
filter_small_age = users_age['age']>5
users_age = users_age[filter_small_age]
#cannot reindex from a duplicate axis
filter_big_age = users_age['age']<100
users_age = users_age[filter_big_age]
#users_age.index = range(len(users_age))＃可以把索引重排
result_age=users_age['age'].describe()
print('age result:')
print(result_age)
plt.figure(figsize=(8,4))
sb.distplot(users_age['age'], rug=True)
plt.show()

users_destination = train_users.replace('NDF',np.nan)
users_destination = users_destination.dropna(subset=['country_destination'])
plt.figure(figsize=(8,4))
sb.countplot(x='country_destination', data=users_destination)
plt.xlabel('country_destination')
plt.ylabel('Number of users')
plt.show()

users.signup_method.unique()
plt.figure(figsize=(8,4))
sb.countplot(x='signup_method', data = users)
plt.xlabel('Signup Method')
plt.ylabel('Number of users')
plt.title('sign up method distribution')
plt.show()

plt.figure(figsize=(8,4))
#一次比較兩條可用hue
sb.countplot(x='country_destination', data=users_destination, hue='signup_app')
plt.xlabel('Destination Country')
plt.ylabel('Number of users')
plt.title('Destination country based on signup app')
plt.legend(loc = 'upper right')
plt.show()


#季節統計
train_users_season = data_encoder(train_users)
users_season = train_users_season[["date_account_created"]]
users_season = users_season.replace(1,'winter')
users_season = users_season.replace(2,'spring')
users_season = users_season.replace(3,'summer')
users_season = users_season.replace(4,'autumn')

plt.figure(figsize=(8,4))
#一次比較兩條可用hue
sb.countplot(x='date_account_created', data=users_season, )
plt.xlabel('season')
plt.ylabel('Number of users')
plt.title('Destination country based on signup app')
plt.legend(loc = 'upper right')
plt.show()

"""
train_data
"""    
train_data = train_users[['gender','age','date_account_created','country_destination']]
train_data = data_encoder(train_data)
train_data = label_encoder(train_data)
train_data = shuffle(train_data)
train_data = train_data.values.tolist()

"""
test_data
"""
test_data = test_users[['gender','age','date_account_created']]
test_data = data_encoder(test_data)
test_data = shuffle(test_data)
test_data = test_data.values.tolist()
