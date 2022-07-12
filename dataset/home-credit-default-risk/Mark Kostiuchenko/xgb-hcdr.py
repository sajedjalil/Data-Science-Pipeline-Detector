#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial
from scipy.stats import kurtosis, iqr, skew
from lightgbm import LGBMClassifier
from sklearn.linlinear_modeldel import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('max_columns', None)


# In[2]:


#Загрузка данных.
train  = pd.read_csv('../input/home-credit-default-risk/application_train.csv') #Базовый треин датасет.
test  = pd.read_csv('../input/home-credit-default-risk/application_test.csv') #Базовый тест датасет.

#---bureau.csv and bureau_balance.csv---
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv') #Данные о предыдущих займах.
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')#Дополняющая таблица с балансом дял bureau.

#---previous_applications.csv---
previous_application = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')#Все предыдущие заявки на получение кредитов в рамках Home Credit.

# ---POS_CASH_balance.csv---
POS_CASH_balance = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')#Снапшоты баланса заявителей.

# ---installments_payments.csv---
installments_payments = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')#История погашения ранее выданных кредитов.

# ---credit_card_balance.csv---
credit_card_balance = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')#Ежемесячные снимки баланса кредитных карт.


# In[3]:


#----------------------------application_train.csv--------------------------

app_train = train.append(test)#Расширяем трейн выборку тест выборкой.

app_train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)#Убираем вбросы и аномалии с поля DAYS_EMPLOYED

#Feature Engineering
app_train['NEW_INCOME_BY_ORGANIZATION'] = app_train[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']#Получение медианного значение дохода конкретных отраслей
app_train['NEW_CREDIT_TO_ANNUITY'] = app_train['AMT_CREDIT'] / app_train['AMT_ANNUITY']#Отношение размера кредита к ежемесячному доходу
app_train['NEW_CREDIT_TO_GOODS'] = app_train['AMT_CREDIT'] / app_train['AMT_GOODS_PRICE']#Отношение кредита к стоимости имущества

app_train['NEW_SOURCES'] = app_train['EXT_SOURCE_1'] * app_train['EXT_SOURCE_2'] * app_train['EXT_SOURCE_3']#Произведение полей EXT_SOURCE 
app_train['NEW_EXT_SOURCES_MEAN'] = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)#Медианное значение поле EXT_SOURCE
app_train['NEW_SOURCES_STD'] = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)#Среднеквадратическое отклонение полей EXT_SOURCE 

#Отношение возраста машины к возрасту заявителя и его трудовому стажу
app_train['NEW_CAR_TO_BIRTH'] = app_train['OWN_CAR_AGE'] / app_train['DAYS_BIRTH']
app_train['NEW_CAR_TO_EMPLOY'] = app_train['OWN_CAR_AGE'] / app_train['DAYS_EMPLOYED']

#Удаляем поля FLAG_DOCUMENT из-за низкой значимости, и в целях ускорить обучение модели
Column_List_To_Drop=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
app_train= app_train.drop(Column_List_To_Drop, axis=1)

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:#Бинарное кодирование для полей CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY
    app_train[bin_feature], uniques = pd.factorize(app_train[bin_feature])


# In[4]:


#----------------------------bureau.csv and bureau_balance.csv--------------------------

previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'PREVIOUS_LOANS_COUNT'})#Общее количество предыдущих кредитов, взятых каждым клиентом
app_train = app_train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')

#Feature Engineering
app_train['NEW_MONTHS_BALANCE_MIN'] = bureau_balance['MONTHS_BALANCE'].min()
app_train['NEW_MONTHS_BALANCE_MAX'] = bureau_balance['MONTHS_BALANCE'].max()
app_train['NEW_MONTHS_BALANCE_SIZE'] = (bureau_balance['MONTHS_BALANCE'].sum()).mean()

bureau['NEW_AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].sum()
bureau['NEW_AMT_CREDIT_SUM_SIZE'] = bureau['NEW_AMT_CREDIT_SUM'].mean()

bureau_bal_mean = bureau_balance.groupby('SK_ID_BUREAU', as_index=False).mean().add_prefix('BUR_BAL_MEAN_')#Нахождение медианого значения всех кредитов для каждого клиента в таблице bureau_balance
bureau_bal_mean = bureau_bal_mean.rename(columns = {'BUR_BAL_MEAN_SK_ID_BUREAU' : 'SK_ID_BUREAU'})
 
bureau = bureau.merge(bureau_bal_mean, on = 'SK_ID_BUREAU', how = 'left')#"Мердж" с bureau
bureau.drop('SK_ID_BUREAU', axis = 1, inplace = True) #Удаление ид поля таблицы bureau_balance

bureau_mean_values = bureau.groupby('SK_ID_CURR', as_index=False).mean().add_prefix('PREV_BUR_MEAN_')#Нахождение медианого значения всех кредитов для каждого клиента в таблице bureau
bureau_mean_values = bureau_mean_values.rename(columns = {'PREV_BUR_MEAN_SK_ID_CURR' : 'SK_ID_CURR'})

app_train = app_train.merge(bureau_mean_values, on = 'SK_ID_CURR', how = 'left')#"Мердж" с главным датасетом


# In[5]:


#Удаление внутрних "id" полей таблиц,так-как в конечном итоге они будут объединены с таблицей previous_applications

credit_card_balance.drop('SK_ID_CURR', axis = 1, inplace = True)
installments_payments.drop('SK_ID_CURR', axis = 1, inplace = True)
POS_CASH_balance.drop('SK_ID_CURR', axis = 1, inplace = True)


# In[6]:


#----------------------------previous_application.csv--------------------------

#Количество предыдущих обращений клиентов к Хоум-кредиту
previous_application_counts = previous_application.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count().rename(columns = {'SK_ID_PREV': 'PREVIOUS_APPLICATION_COUNT'})

app_train = app_train.merge(previous_application_counts, on = 'SK_ID_CURR', how = 'left')#"Мердж" с главным датасетом

#Уборка аномалий
previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)


# In[7]:


#----------------------------credit_card_balance.csv--------------------------

#Нахождение медианого значения всех кредитов балансов каждого клиента в таблице credit_card_balance
credit_card_balance_mean = credit_card_balance.groupby('SK_ID_PREV', as_index=False).mean().add_prefix('CARD_MEAN_')
credit_card_balance_mean = credit_card_balance_mean.rename(columns = {'CARD_MEAN_SK_ID_PREV' : 'SK_ID_PREV'})

previous_application = previous_application.merge(credit_card_balance_mean, on = 'SK_ID_PREV', how = 'left')#"Мердж" с главным датасетом


# In[8]:


#----------------------------installments_payments.csv--------------------------

#Нахождение медианого значения всех расстрочек каждого клиента в таблице installments_payments
install_pay_mean = installments_payments.groupby('SK_ID_PREV', as_index=False).mean().add_prefix('INSTALL_MEAN_')
install_pay_mean = install_pay_mean.rename(columns = {'INSTALL_MEAN_SK_ID_PREV' : 'SK_ID_PREV'})

previous_application = previous_application.merge(install_pay_mean, on = 'SK_ID_PREV', how = 'left')


# In[9]:


#----------------------------POS_CASH_balance.csv--------------------------

#Нахождение медианого значения всех POS-точек каждого клиента в таблице POS_CASH_balance
POS_mean = POS_CASH_balance.groupby('SK_ID_PREV', as_index=False).mean().add_prefix('POS_MEAN_')
POS_mean = POS_mean.rename(columns = {'POS_MEAN_SK_ID_PREV' : 'SK_ID_PREV'})

previous_application = previous_application.merge(POS_mean, on = 'SK_ID_PREV', how = 'left')


# In[10]:


#----------------------------collapse previous_application.csv--------------------------

#Нахождение медианого значения всех предыдущих каждого клиента в таблице previous_application
prev_appl_mean = previous_application.groupby('SK_ID_CURR', as_index=False).mean().add_prefix('PREV_APPL_MEAN_')
prev_appl_mean = prev_appl_mean.rename(columns = {'PREV_APPL_MEAN_SK_ID_CURR' : 'SK_ID_CURR'})

prev_appl_mean = prev_appl_mean.drop('PREV_APPL_MEAN_SK_ID_PREV', axis = 1)


# In[11]:


app_train = app_train.merge(prev_appl_mean, on = 'SK_ID_CURR', how = 'left')#Объеденение всех 4 таблиц с главным датасетом


# In[12]:


#----------------------------Предподготовка данных--------------------------

#Выполняем сплит на тест и трейн выборку по идентификаторам в исходных наборах данных
train1 = app_train[app_train['SK_ID_CURR'].isin(train.SK_ID_CURR)]

test1 = app_train[app_train.SK_ID_CURR.isin(test.SK_ID_CURR)]
test1.drop('TARGET', axis = 1, inplace = True)


# In[13]:


print('Training Features shape with categorical columns: ', train1.shape)
print('Testing Features shape with categorical columns: ', test1.shape)


# In[14]:


#Приминение дамми кодирования
train1 = pd.get_dummies(train1)
test1 = pd.get_dummies(test1)


# In[15]:


# Проверяем шейпы после кодирования
print('Training Features shape with dummy variables: ', train1.shape)
print('Testing Features shape with dummy variables: ', test1.shape)


# In[16]:


TARGET = train1.TARGET #Сохраняем  TARGET переменную
train1.drop('TARGET', axis = 1, inplace = True) #Удаляем TARGET из train1


# In[17]:


#Выравниваем датасеты
train1, test1 = train1.align(test1, join = 'inner', axis = 1)


# In[18]:


print(train1.shape)
print(test1.shape)


# In[19]:


# Для удаления пропущенных значений используем класс Imputer из библиотеки Sklearn с заполнением пропусков при помощи медианных значений
from sklearn.preprocessing import MinMaxScaler, Imputer

imputer = Imputer(strategy = 'median')


# In[20]:


print('Missing values in train data: ', sum(train1.isnull().sum()))
print('Missing values in test data: ', sum(test1.isnull().sum()))


# In[21]:


#Фитим Imputer на train1 датасете
imputer.fit(train1)


# In[22]:


#Заполняем пропущеные значения
imputed_train = imputer.transform(train1)
imputed_test = imputer.transform(test1)


# In[23]:


#В датасете присутствует проблема с дисперсией данных, так как данные слишком сильно колеблятся между собой
#По этому стоит представить каждый столбец в значениях от 0 до 1 в соответвии с их относителными значениями
scaler = MinMaxScaler(feature_range = (0, 1))


# In[24]:


scaler.fit(train1)


# In[25]:


#Применяем маштабирование
scaled_train = scaler.transform(train1)
scaled_test = scaler.transform(test1)


# In[26]:


train1 = pd.DataFrame(scaled_train, index=train1.index, columns=train1.columns)
test1 = pd.DataFrame(scaled_test, index=test1.index, columns=test1.columns)


# In[27]:


#----------------------------------Моделирование------------------------
import xgboost as xgb

X_train = train1
y_train = TARGET

X_test = test1

# Создание объектов DMatrix
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)

# Определение параметров модели

params = {
    'min_child_weight': 19.0,         # минимальное количество окончательно классифицированных записей(классификатор не будет разбивать лист дальше на части, когда размер листа достигнет 19 экземпляров) 
    'objective': 'binary:logistic',   # наша цель-бинарная классификация
    'max_depth': 7,                   # maximum tree depth
    'eta': 0.025,                     # learning rate
    'eval_metric': 'auc',             # обучать модель с учетом конкретной метрики, в нашем случае-AUC
    'max_delta_step': 1.8,          
    'colsample_bytree': 0.4,               
    'subsample': 0.8,
    'gamma': 0.65
    }

# Ватчлист для вывода метрик во время обучения
watchlist = [(dtrain, 'train')]

XGB_model = xgb.train(params, dtrain, 
                300,                  # numrows
                watchlist,            # для контроля процесса обучения
                verbose_eval=50)      # чтобы показать прогресс обучения каждые 50 раундов

# Прогнозирование по тестовым данным
XGB_pred = XGB_model.predict(dtest)


# In[28]:


# Создать файл сабмишена
submission_2 = test[['SK_ID_CURR']]
submission_2['TARGET'] = XGB_pred


# In[29]:


# Експорт файла в csv формат
submission_2.to_csv('XGB_prediciton.csv', index = False)


# In[30]:


importances = XGB_model.get_score()

importances = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importances.sort_values(by = 'Importance', ascending = True,  inplace = True)
importances[importances['Importance'] > 200].plot(kind = 'barh', x = 'Feature', figsize = (8,12), color = 'orange')



