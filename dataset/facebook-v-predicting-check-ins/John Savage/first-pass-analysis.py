
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy

import time
start_time = time.time()

df_train=pd.read_csv('../input/train.csv',encoding='utf8',index_col=0)
df_test=pd.read_csv('../input/test.csv',encoding='utf8',index_col=0)
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

# Find the (half) hour of the day
import math
df_train['time1']=df_train['time'].apply(lambda x:math.floor(x/30) % 48)
df_train.drop('time',inplace=True,axis=1)
df_test['time1']=df_test['time'].apply(lambda x:math.floor(x/30) % 48)
df_test.drop('time',inplace=True,axis=1)
print("--- Time set: %s minutes ---" % round(((time.time() - start_time)/60),2))

# find the closest 6 train points
import sklearn.neighbors as neigh

tree = neigh.KDTree(df_train[['x','y']],leaf_size=2)
print("--- tree built: %s minutes ---" % round(((time.time() - start_time)/60),2))
dist, ind = tree.query(df_test[['x','y']], k=6)
del dist
del tree

print("--- dist set: %s minutes ---" % round(((time.time() - start_time)/60),2))

#make lists for fast lookups, 
time_lookup_test = list(df_test['time1'])
del df_test
place_id_lookup = list(df_train['place_id'].astype(str))
time_lookup_train = list(df_train['time1'])
del df_train
print("--- lookups set: %s minutes ---" % round(((time.time() - start_time)/60),2))

# pick the 3 points closest in time
def dTime(t1,t2,T):
    delta=abs(t2-t1)
    delta=T - delta if delta > 0.5 * T else delta
    return delta
sort_ind=[ sorted(j,key=(lambda t: dTime(time_lookup_train[t],time_lookup_test[i],48)))[:3] for i,j in enumerate(ind)]
#sort_ind=[ j[:3] for i,j in enumerate(ind)]

del ind,time_lookup_test,time_lookup_train
print("--- sorted: %s minutes ---" % round(((time.time() - start_time)/60),2))

#Turn indices to place_ids
res=[ [place_id_lookup[j] for j in i] for i in sort_ind]
del place_id_lookup,sort_ind
print("--- res set: %s minutes ---" % round(((time.time() - start_time)/60),2))

#print out dataframe
df_out=pd.DataFrame(columns=['row_id','place_id'])
df_out['place_id']=[ i[0]+' '+i[1]+' '+i[2] for i in res]
del res
df_out['row_id']=range(len(df_out['place_id']))    
#df_out.head()
df_out.to_csv('submission5.csv',index=False)
print("--- done: %s minutes ---" % round(((time.time() - start_time)/60),2))

