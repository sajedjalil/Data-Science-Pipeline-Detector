
import sys
import numpy as np
import pandas as pd

sessions= pd.read_csv('../input/sessions.csv')

print(sessions.head())
print(sessions.info())
print(sessions.apply(lambda x: x.nunique(),axis=0))

#sessions['action'] = sessions['action'].fillna('999')
#data roll-up
#secs_elapsed
grpby = sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grpby.columns = ['user_id','secs_elapsed']

# agg = grpby['secs_elapsed'].agg({'time_spent' : np.sum})

#action
#print(sessions.action_type.value_counts())
#print(sessions.groupby(['action_type'])['user_id'].nunique().reset_index())
action_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
action_type = action_type.drop(['booking_response'],axis=1)
#print(action_type.head())

#print(sessions.groupby(['device_type'])['user_id'].nunique().reset_index())
#print(sessions.groupby(['user_id'])['device_type'].nunique().reset_index())
device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
device_type = device_type.drop(['Blackberry','Opera Phone','iPodtouch','Windows Phone'],axis=1)
#device_type = device_type.replace(device_type.iloc[:,1:]>0,1)
print(device_type.info())

sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')

sessions_data = pd.merge(sessions_data,grpby,on='user_id',how='inner')
print(sessions_data.head())

