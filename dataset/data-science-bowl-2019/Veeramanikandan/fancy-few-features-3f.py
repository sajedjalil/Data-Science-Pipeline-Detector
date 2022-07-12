# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
import seaborn as sns
from catboost import CatBoostRegressor,CatBoostClassifier
from matplotlib import pyplot
import shap
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from time import time
from tqdm import tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import json

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode


# %% [code]
os.listdir('../input/data-science-bowl-2019')
#os.listdir('../input/')

keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count',
             'event_code','title' ,'game_time', 'type', 'world','timestamp']
train=pd.read_csv('../input/data-science-bowl-2019/train.csv',usecols=keep_cols)
#train=pd.read_csv('../input/train.csv',usecols=keep_cols)
train_labels=pd.read_csv('../input/data-science-bowl-2019/train_labels.csv',
                         usecols=['installation_id','game_session','accuracy_group'])
test=pd.read_csv('../input/data-science-bowl-2019/test.csv',usecols=keep_cols)
submission=pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

# %% [code]
print(train.shape,train_labels.shape)
x=train_labels['accuracy_group'].value_counts()
sns.barplot(x.index,x)

# %% [code]
not_req=(set(train.installation_id.unique()) - set(train_labels.installation_id.unique()))
#print(not_req)

# %% [code]
train_new=~train['installation_id'].isin(not_req)
train.where(train_new,inplace=True)
train.dropna(inplace=True)
train['event_code']=train.event_code.astype(int)

# %% [code]
def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear
    return df

# %% [code]
time_features=['month','hour','year','dayofweek','weekofyear']
def prepare_data(df):
    df=extract_time_features(df)    
    #df=df.drop('timestamp',axis=1)
    #df['timestamp']=pd.to_datetime(df['timestamp'])
    df['hour_of_day']=df['timestamp'].map(lambda x : int(x.hour))

    join_one=pd.get_dummies(df[['event_code','installation_id','game_session']],
                            columns=['event_code']).groupby(['installation_id','game_session'],
                                                            as_index=False,sort=False).agg(sum)

    agg={'event_count':sum,'game_time':['sum','mean'],'event_id':'count'}

    join_two=df.drop(time_features,axis=1).groupby(['installation_id','game_session']
                                                   ,as_index=False,sort=False).agg(agg)
    
    join_two.columns= [' '.join(col).strip() for col in join_two.columns.values]
    

    join_three=df[['installation_id','game_session','type','world','title']].groupby(
                ['installation_id','game_session'],as_index=False,sort=False).first()
    
    join_four=df[time_features+['installation_id','game_session']].groupby(['installation_id',
                'game_session'],as_index=False,sort=False).agg(mode)[time_features].applymap(lambda x: x.mode[0])
    
    join_one=join_one.join(join_four)
    
    join_five=(join_one.join(join_two.drop(['installation_id','game_session'],axis=1))). \
                        join(join_three.drop(['installation_id','game_session'],axis=1))
    
    return join_five



# %% [code]
join_train=prepare_data(train)
cols=join_train.columns.to_list()[2:-3]
join_train[cols]=join_train[cols].astype('int16')

# %% [code]
join_test=prepare_data(test)
cols=join_test.columns.to_list()[2:-3]
join_test[cols]=join_test[cols].astype('int16')

# %% [code]
cols=join_test.columns[2:-12].to_list()
cols.append('event_id count')
cols.append('installation_id')

# %% [code]
df=join_test[['event_count sum','game_time mean','game_time sum',
    'installation_id']].groupby('installation_id',as_index=False,sort=False).agg('mean')

df_two=join_test[cols].groupby('installation_id',as_index=False,
                               sort=False).agg('sum').drop('installation_id',axis=1)

df_three=join_test[['title','type','world','installation_id']].groupby('installation_id',
         as_index=False,sort=False).last().drop('installation_id',axis=1)

df_four=join_test[time_features+['installation_id']].groupby('installation_id',as_index=False,sort=False). \
        agg(mode)[time_features].applymap(lambda x : x.mode[0])

# %% [code]
final_train=pd.merge(train_labels,join_train,on=['installation_id','game_session'],
                                         how='left').drop(['game_session'],axis=1)

#final_test=join_test.groupby('installation_id',as_index=False,sort=False).last().drop(['game_session','installation_id'],axis=1)
final_test=(df.join(df_two)).join(df_three.join(df_four)).drop('installation_id',axis=1)

# %% [code]
df=final_train[['event_count sum','game_time mean','game_time sum','installation_id']]. \
    groupby('installation_id',as_index=False,sort=False).agg('mean')

df_two=final_train[cols].groupby('installation_id',as_index=False,
                                 sort=False).agg('sum').drop('installation_id',axis=1)

df_three=final_train[['accuracy_group','title','type','world','installation_id']]. \
        groupby('installation_id',as_index=False,sort=False). \
        last().drop('installation_id',axis=1)

df_four=join_train[time_features+['installation_id']].groupby('installation_id',as_index=False,sort=False). \
        agg(mode)[time_features].applymap(lambda x : x.mode[0])



final_train=(df.join(df_two)).join(df_three.join(df_four)).drop('installation_id',axis=1)

# %% [code]
print(final_train.shape,final_test.shape)
#print(len(set(final_train.columns) & set(final_test.columns)))

# %% [code]
final=pd.concat([final_train,final_test])
encoding=['type','world','title']
for col in encoding:
    lb=LabelEncoder()
    lb.fit(final[col])
    final[col]=lb.transform(final[col])
final_train=final[:len(final_train)]
final_test=final[len(final_train):]

# %% [code]
X=final_train.drop('accuracy_group',axis=1)
X1=final_test.drop('accuracy_group',axis=1)
Y=final_train['accuracy_group']

# %% [code]
#Cat Model creation
from catboost import CatBoostRegressor,CatBoostClassifier
cat_clf=CatBoostClassifier(loss_function= 'MultiClass',task_type= "CPU",iterations= 200,
                   od_type= "Iter",depth= 10,colsample_bylevel= 0.5,early_stopping_rounds= 100,
                   l2_leaf_reg= 18,random_seed= 42,use_best_model= True)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

scores=[]
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    cat_clf.fit(x_train,y_train,eval_set=(x_test, y_test),verbose=100)
    pred_val=cat_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
    
print('Mean score:',np.mean(scores))

# %% [code]
#xg model
xgb_clf=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.05, max_delta_step=0, max_depth=10,
              min_child_weight=6, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scores=[]
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    xgb_clf.fit(x_train,y_train)
    pred_val=xgb_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#randomforest
from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=10, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
score=[]
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    rf_clf.fit(x_train,y_train)
    pred_val=rf_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#KNN
from sklearn.neighbors import KNeighborsClassifier
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=9)
score=[]
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    knn_clf.fit(x_train,y_train)
    pred_val=knn_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#SVM
from sklearn import svm
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
svm_clf = svm.SVC(probability=True)
score=[]
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    svm_clf.fit(x_train,y_train)
    pred_val=svm_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#naive_bayes
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
score=[]
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    gnb_clf.fit(x_train,y_train)
    pred_val=gnb_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#Decision tree
from sklearn import tree
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
dt_clf = tree.DecisionTreeClassifier()
score=[]
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    dt_clf.fit(x_train,y_train)
    pred_val=dt_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#LR
from sklearn.linear_model import LogisticRegression
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
reg_clf = LogisticRegression(random_state=0)
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    reg_clf.fit(x_train,y_train)
    pred_val=reg_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
lda_clf = LinearDiscriminantAnalysis()
score=[]
for train_idx, val_idx in cv.split(X,Y):
    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 
    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     
    lda_clf.fit(x_train,y_train)
    pred_val=lda_clf.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#save models
import pickle
def save_models():
    pickle.dump(cat_clf, open('models/cat.sav', 'wb'))
    pickle.dump(xgb_clf, open('models/xgb.sav', 'wb'))
    pickle.dump(rf_clf, open('models/rf.sav', 'wb'))
    pickle.dump(knn_clf, open('models/knn.sav', 'wb'))
    pickle.dump(svm_clf, open('models/svm.sav', 'wb'))
    pickle.dump(gnb_clf, open('models/gnb.sav','wb'))
    pickle.dump(dt_clf, open('models/dt.sav', 'wb'))
    pickle.dump(reg_clf, open('models/lr.sav', 'wb'))
    pickle.dump(lda_clf, open('models/lda.sav', 'wb'))
    
def load_models():
    m1= open("models/cat.sav","rb")
    clf1= pickle.load(m1)
    
    m2= open("models/xgb.sav","rb")
    clf2= pickle.load(m2)
    
    m3= open("models/rf.sav","rb")
    clf3= pickle.load(m3)
    
    m4= open("models/knn.sav","rb")
    clf4= pickle.load(m4)
    
    m5= open("models/svm.sav","rb")
    clf5= pickle.load(m5)
    
    m6= open("models/gnb.sav","rb")
    clf6= pickle.load(m6)
    
    m7= open("models/dt.sav","rb")
    clf7= pickle.load(m7)
    
    m8= open("models/lr.sav","rb")
    clf8= pickle.load(m8)
    
    m9= open("models/lda.sav","rb")
    clf9= pickle.load(m9)
    
    return clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9


# %% [code]
#save models 
#save_models()
#load models
#cat_clf,xgb_clf,rf_clf,knn_clf,svm_clf,gnb_clf,dt_clf,reg_clf,lda_clf=load_models()

# %% [code]
def collect_data_level_one(X,X1):    
    #train data
    tr1=cat_clf.predict_proba(X)
    tr2=xgb_clf.predict_proba(X)
    tr3=rf_clf.predict_proba(X)
    tr4=knn_clf.predict_proba(X)
    tr5=svm_clf.predict_proba(X)
    tr6=gnb_clf.predict_proba(X)
    tr7=dt_clf.predict_proba(X)
    tr8=reg_clf.predict_proba(X)
    tr9=lda_clf.predict_proba(X)
    meta_data=np.column_stack((tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9))
    
    #test data
    ts1=cat_clf.predict_proba(X1)
    ts2=xgb_clf.predict_proba(X1)
    ts3=rf_clf.predict_proba(X1)
    ts4=knn_clf.predict_proba(X1)
    ts5=svm_clf.predict_proba(X1)
    ts6=gnb_clf.predict_proba(X1)
    ts7=dt_clf.predict_proba(X1)
    ts8=reg_clf.predict_proba(X1)
    ts9=lda_clf.predict_proba(X1)
    meta_test_data=np.column_stack((ts1,ts2,ts3,ts4,ts5,ts6,ts7,ts8,ts9))
    
    return meta_data,meta_test_data 

# %% [code]
#meta data prepration
tr,ts=collect_data_level_one(X,X1)  

# %% [code]
len(tr)

# %% [code]
#meta one XGB
import xgboost as xgb
mx1=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.05, max_delta_step=0, max_depth=10,
                  min_child_weight=6, missing=None, n_estimators=200, n_jobs=1,
                  nthread=None, objective='multi:softprob', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

score=[]
for train_idx, val_idx in cv.split(tr,Y):
    x_train, y_train = tr[train_idx], Y[train_idx] 
    x_test, y_test = tr[val_idx], Y[val_idx]     
    mx1.fit(x_train,y_train)
    pred_val=mx1.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#meta two RF
from sklearn.ensemble import RandomForestClassifier
mx2=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=10, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
score=[]
for train_idx, val_idx in cv.split(tr,Y):
    x_train, y_train = tr[train_idx], Y[train_idx] 
    x_test, y_test = tr[val_idx], Y[val_idx]   
    mx2.fit(x_train,y_train)
    pred_val=mx2.predict(x_test)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
print('Mean score:',np.mean(scores))

# %% [code]
#meta three CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.backend.clear_session() ## For easy reset of notebook state
inputs = keras.Input(shape=(36,), name='ip')
x = layers.Dense(72, activation='sigmoid', name='dense_1')(inputs)
x = layers.Dense(64, activation='sigmoid', name='dense_1')(x)
x = layers.Dense(32, activation='sigmoid', name='dense_1')(x)
x = layers.Dense(24, activation='tanh', name='dense_4')(x)
x = layers.Dense(12, activation='relu', name='dense_7')(x)
x = layers.Dropout(.25)(inputs)
x = layers.Dense(8, activation='relu', name='dense_9')(x)
outputs = layers.Dense(4, activation='softmax', name='predictions')(x)
mx3 = keras.Model(inputs=inputs, outputs=outputs)
mx3.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(), # Loss function to minimize
              metrics=[keras.metrics.SparseCategoricalAccuracy()]) # List of metrics to monitor
#data split up
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
score=[]
for train_idx, val_idx in cv.split(tr,Y):
    x_train, y_train = tr[train_idx], Y[train_idx] 
    x_test, y_test = tr[val_idx], Y[val_idx]   
    history = mx3.fit(x_train, y_train,validation_data=(x_test, y_test),batch_size=24,epochs=50)
    #print('\nhistory dict:', history.history)
    pred_val=np.argmax(mx3.predict(x_test),axis=1)
    score=cohen_kappa_score(pred_val,y_test,weights='quadratic')
    print('choen_kappa_score :',score)
    scores.append(score)
    
print('Mean score:',np.mean(scores))
    

# %% [code]
#save models
import pickle
def save_meta_models():
    pickle.dump(mx1, open('models/mx1.sav', 'wb'))
    pickle.dump(mx2, open('models/mx2.sav', 'wb'))
    pickle.dump(mx3, open('models/mx3.sav', 'wb'))
    
def load_meta_models():
    mx1= open("models/mx1.sav","rb")
    clf1= pickle.load(mx1)
    
    mx2= open("models/mx2.sav","rb")
    clf2= pickle.load(mx2)
    
    mx3= open("models/mx3.sav","rb")
    clf3= pickle.load(mx3)    
      
    return clf1,clf2,clf3

# %% [code]
#save_meta_models()

# %% [code]
def collect_data_level_two(tr,ts):    
    #train data
    mtr1=mx1.predict_proba(tr)
    mtr2=mx2.predict_proba(tr)
    mtr3=mx3.predict(tr)   
    mtr=np.column_stack((mtr1,mtr2,mtr3))
    
    #test data
    mts1=mx1.predict_proba(ts)
    mts2=mx2.predict_proba(ts)
    mts3=mx3.predict(ts)
    mts=np.column_stack((mts1,mts2,mts3))
    
    return mtr,mts 

def save_data_csv():
    X.to_csv('models/data/rtrain.csv')
    X1.to_csv('models/data/rtest.csv')
    np.savetxt('models/data/meta_tr.csv', tr, delimiter=",")
    np.savetxt('models/data/meta_ts.csv', ts, delimiter=",")
    np.savetxt('models/data/m_tr.csv', mtr, delimiter=",")
    np.savetxt('models/data/m_ts.csv', mts, delimiter=",")

# %% [code]
#build the final Meta data
mtr,mts=collect_data_level_two(tr,ts)

# %% [code]
# Save all the processed data into csv
#save_data_csv()

# %% [code]
#meta model XGB Boosting
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
meta=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.05, max_delta_step=0, max_depth=10,
                  min_child_weight=6, missing=None, n_estimators=200, n_jobs=1,
                  nthread=None, objective='multi:softprob', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
score=[]
for train_idx, val_idx in cv.split(mtr,Y):
    x_train, y_train = mtr[train_idx], Y[train_idx] 
    x_test, y_test = mtr[val_idx], Y[val_idx]     
    meta.fit(x_train,y_train)
    f_pred=meta.predict(x_test)
    score=cohen_kappa_score(f_pred,y_test,weights='quadratic')
    print('cohen_kappa_score :',score)
    scores.append(score)
    
print('Mean score:',np.mean(scores))

# %% [code]
#predict for the test data
final_pred=meta.predict(mts)
#sub=pd.DataFrame({'installation_id':submission.installation_id,'accuracy_group':final_pred})
#sub.to_csv('submission.csv',index=False)
#sub.groupby(['accuracy_group']).count()

submission['accuracy_group'] = final_pred.astype(int)
submission.to_csv('submission.csv', index=False)
submission.groupby(['accuracy_group']).count()

