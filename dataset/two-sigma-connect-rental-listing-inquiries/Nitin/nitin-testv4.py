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

import json  
import zipfile  
import matplotlib.pyplot as plt

path= "../input/"
d = None  
data = None  

with open(path+"train.json") as f:
    #data = f.read()
    d=json.load(f)
    f.close()

type(d)


df=pd.DataFrame(d)
len(df)
df.head(10)

df['created'] = df['created'].astype('datetime64[ns]')
a=max(df['created'])-df['created']
df['daysFromCreated']=a/np.timedelta64(1, 'D')
xl=df.longitude
yl=df.latitude
xl=(xl-np.mean(xl))/np.std(xl)
yl=(yl-np.mean(yl))/np.std(yl)

plt.plot(xl,yl,'ro')
plt.show()

from sklearn.cluster import KMeans

a= np.array(xl)
b=np.array(yl)
c=np.column_stack((a,b))
kmeans = KMeans(n_clusters=20)
kmeans.fit_predict(c)
centers = kmeans.cluster_centers_
prediction = kmeans.predict(c)
plt.scatter(centers[:, 0], centers[:, 1], marker='x')

plt.show()
df["KMeans_Clusters"]= prediction
df.groupby("KMeans_Clusters").count()

cl_count = df.groupby("KMeans_Clusters").size()
type(cl_count)

toBeRemovedCls=[]
for ind in range (len(cl_count)):
    if cl_count[ind]<1000 :
        toBeRemovedCls.append(ind)
        

df['KMeans_Clusters']=df['KMeans_Clusters'].replace(toBeRemovedCls, len(cl_count)+1)



##Test

d_test = None
#data_test=None
with open(path+"test.json") as f1:
    #data = f.read()
    d_test=json.load(f1)
    f1.close()

type(d_test)
df_test=pd.DataFrame(d_test)
len(df_test)

##Test

df_test['created'] = df_test['created'].astype('datetime64[ns]')
#a=max(df_test['created'])-df_test['created']
df_test['daysFromCreated']=(max(df_test['created'])-df_test['created'])/np.timedelta64(1, 'D')
xl_test=df_test.longitude
yl_test=df_test.latitude
xl_test=(xl_test-np.mean(xl_test))/np.std(xl_test)
yl_test=(yl_test-np.mean(yl_test))/np.std(yl_test)
a_test= np.array(xl_test)
b_test=np.array(yl_test)
c_test=np.column_stack((a_test,b_test))
kmeans.fit_predict(c_test)
prediction_test = kmeans.predict(c_test)

df_test["KMeans_Clusters"]= prediction_test

df_test['KMeans_Clusters']=df_test['KMeans_Clusters'].replace(toBeRemovedCls, len(cl_count)+1)

df_test['KMeans_Clusters']=df_test['KMeans_Clusters'].astype(object)

################

df['KMeans_Clusters']=df['KMeans_Clusters'].astype(object)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .90

train, test = df[df['is_train']==True], df[df['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
x=[0,1,8,10,13,15,16]
features = df.columns[x]

features
y = pd.factorize(train['interest_level'])[0]
y
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=250,max_depth=15)

# Train the classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], train['interest_level'])
preds=clf.predict(test[features])
clf.predict_proba(test[features])[0:10]
preds[0:5]
test['interest_level'][0:5]
pd.crosstab(test['interest_level'], preds, rownames=['Actual interest_level'], colnames=['Predicted interest_level'])
import sklearn
cv_logloss=sklearn.metrics.log_loss(test['interest_level'], clf.predict_proba(test[features]))

print(cv_logloss)

pred_actualTest = clf.predict_proba(df_test[features])
pred_actualTest
your_permutation = [0,2,1]
i = np.argsort(your_permutation)
i
pred_actualTest1=pred_actualTest[:,i]
out_df = pd.DataFrame(pred_actualTest1)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = df_test.listing_id.values
out_df.to_csv("first1.csv", index=False)

