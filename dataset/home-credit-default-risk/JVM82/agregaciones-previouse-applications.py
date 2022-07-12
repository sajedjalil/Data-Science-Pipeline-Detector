# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/previous_application.csv')
data1=data.groupby(['SK_ID_CURR'],as_index=False).mean()
data_cust1=data.loc[:,['SK_ID_PREV','SK_ID_CURR']]
data_cust=data1.loc[:,['SK_ID_CURR']]
data1=data1.iloc[:,1:]
data1=data1.add_suffix('_mns')
data1=data_cust.join(data1)
data4=data.groupby(['SK_ID_CURR'],as_index=False).median()
data4=data4.iloc[:,1:]
data4=data4.add_suffix('_mdn')
data4=data_cust.join(data4)
data5=data.groupby(['SK_ID_CURR'],as_index=False).std()
data5=data5.iloc[:,1:]
data5=data5.add_suffix('_std')
data5=data_cust.join(data5)
data6=data.groupby(['SK_ID_CURR'],as_index=False).var()
data6=data6.iloc[:,1:]
data6=data6.add_suffix('_var')
data6=data_cust.join(data6)
data7=data.groupby(['SK_ID_CURR'],as_index=False).sum()
data7=data7.iloc[:,1:]
data7=data7.add_suffix('_sum')
data7=data_cust.join(data7)
data8=data.groupby(['SK_ID_CURR'],as_index=False).count()
data8=data8.iloc[:,1:]
data8=data8.add_suffix('_cnt')
data8=data_cust.join(data8)
# data1.head
data_dummy=pd.get_dummies(data.loc[:,['NAME_CONTRACT_TYPE','WEEKDAY_APPR_PROCESS_START','NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_TYPE_SUITE','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION']])
#data.dropna(axis=1, how='all')
data.fillna(data.groupby("SK_ID_CURR").transform("mean"), inplace=True)
data1b=data.groupby(['SK_ID_CURR'],as_index=False).mean()
data1b=data1b.iloc[:,1:]
data1b=data1b.add_suffix('_mns')
data1b=data_cust.join(data1b)
data4b=data.groupby(['SK_ID_CURR'],as_index=False).median()
data4b=data4b.iloc[:,1:]
data4b=data4b.add_suffix('_mdn')
data4b=data_cust.join(data4b)
data5b=data.groupby(['SK_ID_CURR'],as_index=False).std()
data5b=data5b.iloc[:,1:]
data5b=data5b.add_suffix('_std')
data5b=data_cust.join(data5b)
data6b=data.groupby(['SK_ID_CURR'],as_index=False).var()
data6b=data6b.iloc[:,1:]
data6b=data6b.add_suffix('_var')
data6b=data_cust.join(data6b)
data7b=data.groupby(['SK_ID_CURR'],as_index=False).sum()
data7b=data7b.iloc[:,1:]
data7b=data7b.add_suffix('_sum')
data7b=data_cust.join(data7b)
#data7b.head

data_cust1=data_cust1.join(data_dummy)
#data_cust.head
#list(data_cust1.columns.values)
# data_dm_grp=data_cust1.groupby(['SK_ID_CURR'],as_index=False)['NAME_CONTRACT_TYPE_Cash loans','NAME_CONTRACT_TYPE_Consumer loans','NAME_CONTRACT_TYPE_Revolving loans','NAME_CONTRACT_TYPE_XNA','WEEKDAY_APPR_PROCESS_START_FRIDAY','WEEKDAY_APPR_PROCESS_START_MONDAY','WEEKDAY_APPR_PROCESS_START_SATURDAY','WEEKDAY_APPR_PROCESS_START_SUNDAY','WEEKDAY_APPR_PROCESS_START_THURSDAY','WEEKDAY_APPR_PROCESS_START_TUESDAY','WEEKDAY_APPR_PROCESS_START_WEDNESDAY','NAME_CASH_LOAN_PURPOSE_Building a house or an annex','NAME_CASH_LOAN_PURPOSE_Business development','NAME_CASH_LOAN_PURPOSE_Buying a garage','NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land','NAME_CASH_LOAN_PURPOSE_Buying a home','NAME_CASH_LOAN_PURPOSE_Buying a new car','NAME_CASH_LOAN_PURPOSE_Buying a used car','NAME_CASH_LOAN_PURPOSE_Car repairs','NAME_CASH_LOAN_PURPOSE_Education','NAME_CASH_LOAN_PURPOSE_Everyday expenses','NAME_CASH_LOAN_PURPOSE_Furniture','NAME_CASH_LOAN_PURPOSE_Gasification / water supply','NAME_CASH_LOAN_PURPOSE_Hobby','NAME_CASH_LOAN_PURPOSE_Journey','NAME_CASH_LOAN_PURPOSE_Medicine','NAME_CASH_LOAN_PURPOSE_Money for a third person','NAME_CASH_LOAN_PURPOSE_Other','NAME_CASH_LOAN_PURPOSE_Payments on other loans','NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment','NAME_CASH_LOAN_PURPOSE_Refusal to name the goal','NAME_CASH_LOAN_PURPOSE_Repairs','NAME_CASH_LOAN_PURPOSE_Urgent needs','NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday','NAME_CASH_LOAN_PURPOSE_XAP','NAME_CASH_LOAN_PURPOSE_XNA','NAME_CONTRACT_STATUS_Approved','NAME_CONTRACT_STATUS_Canceled','NAME_CONTRACT_STATUS_Refused','NAME_CONTRACT_STATUS_Unused offer','NAME_PAYMENT_TYPE_Cash through the bank','NAME_PAYMENT_TYPE_Cashless from the account of the employer','NAME_PAYMENT_TYPE_Non-cash from your account','NAME_PAYMENT_TYPE_XNA','CODE_REJECT_REASON_CLIENT','CODE_REJECT_REASON_HC','CODE_REJECT_REASON_LIMIT','CODE_REJECT_REASON_SCO','CODE_REJECT_REASON_SCOFR','CODE_REJECT_REASON_SYSTEM','CODE_REJECT_REASON_VERIF','CODE_REJECT_REASON_XAP','CODE_REJECT_REASON_XNA','NAME_TYPE_SUITE_Children','NAME_TYPE_SUITE_Family','NAME_TYPE_SUITE_Group of people','NAME_TYPE_SUITE_Other_A','NAME_TYPE_SUITE_Other_B','NAME_TYPE_SUITE_Spouse, partner','NAME_TYPE_SUITE_Unaccompanied','NAME_CLIENT_TYPE_New','NAME_CLIENT_TYPE_Refreshed','NAME_CLIENT_TYPE_Repeater','NAME_CLIENT_TYPE_XNA','NAME_GOODS_CATEGORY_Additional Service','NAME_GOODS_CATEGORY_Animals','NAME_GOODS_CATEGORY_Audio/Video','NAME_GOODS_CATEGORY_Auto Accessories','NAME_GOODS_CATEGORY_Clothing and Accessories','NAME_GOODS_CATEGORY_Computers','NAME_GOODS_CATEGORY_Construction Materials','NAME_GOODS_CATEGORY_Consumer Electronics','NAME_GOODS_CATEGORY_Direct Sales','NAME_GOODS_CATEGORY_Education','NAME_GOODS_CATEGORY_Fitness','NAME_GOODS_CATEGORY_Furniture','NAME_GOODS_CATEGORY_Gardening','NAME_GOODS_CATEGORY_Homewares','NAME_GOODS_CATEGORY_House Construction','NAME_GOODS_CATEGORY_Insurance','NAME_GOODS_CATEGORY_Jewelry','NAME_GOODS_CATEGORY_Medical Supplies','NAME_GOODS_CATEGORY_Medicine','NAME_GOODS_CATEGORY_Mobile','NAME_GOODS_CATEGORY_Office Appliances','NAME_GOODS_CATEGORY_Other','NAME_GOODS_CATEGORY_Photo / Cinema Equipment','NAME_GOODS_CATEGORY_Sport and Leisure','NAME_GOODS_CATEGORY_Tourism','NAME_GOODS_CATEGORY_Vehicles','NAME_GOODS_CATEGORY_Weapon','NAME_GOODS_CATEGORY_XNA','NAME_PORTFOLIO_Cards','NAME_PORTFOLIO_Cars','NAME_PORTFOLIO_Cash','NAME_PORTFOLIO_POS','NAME_PORTFOLIO_XNA','NAME_PRODUCT_TYPE_XNA','NAME_PRODUCT_TYPE_walk-in','NAME_PRODUCT_TYPE_x-sell','CHANNEL_TYPE_AP+ (Cash loan)','CHANNEL_TYPE_Car dealer','CHANNEL_TYPE_Channel of corporate sales','CHANNEL_TYPE_Contact center','CHANNEL_TYPE_Country-wide','CHANNEL_TYPE_Credit and cash offices','CHANNEL_TYPE_Regional / Local','CHANNEL_TYPE_Stone','NAME_SELLER_INDUSTRY_Auto technology','NAME_SELLER_INDUSTRY_Clothing','NAME_SELLER_INDUSTRY_Connectivity','NAME_SELLER_INDUSTRY_Construction','NAME_SELLER_INDUSTRY_Consumer electronics','NAME_SELLER_INDUSTRY_Furniture','NAME_SELLER_INDUSTRY_Industry','NAME_SELLER_INDUSTRY_Jewelry','NAME_SELLER_INDUSTRY_MLM partners','NAME_SELLER_INDUSTRY_Tourism','NAME_SELLER_INDUSTRY_XNA','NAME_YIELD_GROUP_XNA','NAME_YIELD_GROUP_high','NAME_YIELD_GROUP_low_action','NAME_YIELD_GROUP_low_normal','NAME_YIELD_GROUP_middle','PRODUCT_COMBINATION_Card Street','PRODUCT_COMBINATION_Card X-Sell','PRODUCT_COMBINATION_Cash','PRODUCT_COMBINATION_Cash Street: high','PRODUCT_COMBINATION_Cash Street: low','PRODUCT_COMBINATION_Cash Street: middle','PRODUCT_COMBINATION_Cash X-Sell: high','PRODUCT_COMBINATION_Cash X-Sell: low','PRODUCT_COMBINATION_Cash X-Sell: middle','PRODUCT_COMBINATION_POS household with interest','PRODUCT_COMBINATION_POS household without interest','PRODUCT_COMBINATION_POS industry with interest','PRODUCT_COMBINATION_POS industry without interest','PRODUCT_COMBINATION_POS mobile with interest','PRODUCT_COMBINATION_POS mobile without interest','PRODUCT_COMBINATION_POS other with interest','PRODUCT_COMBINATION_POS others without interest'].agg(lambda x:x.value_counts().index[0])
data8b=data_cust1.groupby(['SK_ID_CURR'],as_index=False).sum()
data8b=data8b.iloc[:,1:]
data8b=data8b.add_suffix('_cnt')
data8b=data_cust.join(data8b)

result = pd.merge(data_cust, data1, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data1, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data4, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data5, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data4, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data6, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data7, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data8, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data1b, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data4b, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data5b, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data4b, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data6b, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data7b, how='left', on=['SK_ID_CURR'])
result = pd.merge(result, data8b, how='left', on=['SK_ID_CURR'])
# result = pd.merge(result, data_dm_grp, how='left', on=['SK_ID_CURR'])
result.to_csv('previous_application_nv1.csv', index=False)
#data_cust1.to_csv('data_dm.csv', put index=False)
