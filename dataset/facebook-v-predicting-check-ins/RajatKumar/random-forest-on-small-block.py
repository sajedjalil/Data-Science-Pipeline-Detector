## Idea developed from 
## "Alexandru Papiu"'s Random Forest on a Few Blocks"


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


## We implement the RF on a small Block, say 4.0<x<4.2, 4.0<y<4.2.
## However, we will have a lareger training block than testing block
## This can make the prediction better, with a better prediction near boundary.

df_train=df_train[(df_train.x>=3.9)&(df_train.x<4.3)&(df_train.y>=3.9)&(df_train.y<4.3)]
#df_train=df_train[(df_train.x>=4.0)&(df_train.x<4.2)&(df_train.y>=4.0)&(df_train.y<4.2)]
df_test=df_test[(df_test.x>=4.0)&(df_test.x<4.2)&(df_test.y>=4.0)&(df_test.y<4.2)]

## Generate new feature
## 6.hour 7.day 8.month 9.year 
time_shift=0
df_train = df_train.sort(['time'])
df_train['hour'] = pd.Series(((df_train.time+time_shift)/60) % 24,index=df_train.index)
df_train['day'] = pd.Series(((df_train.time+time_shift)/1440)%7, index=df_train.index)
df_train['month'] = pd.Series(((df_train.time+time_shift)/1440/30)%12, index=df_train.index)
df_train['year'] = pd.Series(((df_train.time+time_shift)/1440/365), index=df_train.index)


nCol,nRow=df_train.shape
print(nCol,nRow)
size_train_set = int(nCol/10*9)
size_test_set = nCol-size_train_set

x_val=df_train.iloc[size_train_set:,[1,2,3,6,7,8,9]]
y_val=df_train.iloc[size_train_set:,5]
x_train=df_train.iloc[:size_train_set,[1,2,3,6,7,8,9]]
y_train=df_train.iloc[:size_train_set,5]

#x_train=pd.concat([x_train,x_train_extra,x_train_extra2],ignore_index=True)
#y_train=pd.concat([y_train,y_train_extra,y_train_extra],ignore_index=True)

for depth in range(13,14):
    rfc = RandomForestClassifier(n_estimators=30,max_depth=depth)
    rfc.fit(x_train, y_train)
    rfc_y_pred = rfc.set_params( n_jobs = 1 ).predict_proba(x_val)
    idx=np.argmax(rfc_y_pred,1)
    rfc_y_pred1 = rfc.classes_[idx]
    for i in range(size_test_set):
        rfc_y_pred[i,idx[i]]=0
    idx=np.argmax(rfc_y_pred,1)
    rfc_y_pred2 = rfc.classes_[idx]
    for i in range(size_test_set):
        rfc_y_pred[i,idx[i]]=0
    idx=np.argmax(rfc_y_pred,1)
    rfc_y_pred3 = rfc.classes_[idx]

    print("depth:", depth, "time_shift:", time_shift)
    print("E_val_1: ", rfc.score(x_val, y_val))
    #print "E_in_1:  ", rfc.score(x_train, y_train)
    print("E_val_MAP3:  ", accuracy_score(y_val, rfc_y_pred1)+accuracy_score(y_val, rfc_y_pred2)/2+accuracy_score(y_val, rfc_y_pred3)/3)









