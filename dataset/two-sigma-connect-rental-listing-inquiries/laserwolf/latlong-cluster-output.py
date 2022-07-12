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

pd.options.mode.chained_assignment = None

train=pd.read_json("../input/train.json")
test=pd.read_json("../input/test.json")
train["Source"]='train'
test["Source"]='test'
data=pd.concat([train, test]) 


#I use Birch because of how fast it is. 
from sklearn.cluster import Birch
def cluster_latlon(n_clusters, data):  
    #split the data between "around NYC" and "other locations" basically our first two clusters 
    data_c=data[(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]
    data_e=data[~((data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9))]
    #put it in matrix form
    coords=data_c.as_matrix(columns=['latitude', "longitude"])
    
    brc = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)

    brc.fit(coords)
    clusters=brc.predict(coords)
    data_c["cluster_"+str(n_clusters)]=clusters
    data_e["cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings 
    data=pd.concat([data_c,data_e])
    return data 
data=cluster_latlon(13, data)

data["created"]=pd.to_datetime(data["created"])
data["created_month"]=data["created"].dt.month
data["created_day"]=data["created"].dt.day
data["created_hour"]=data["created"].dt.hour

data["num_photos"]=data["photos"].apply(len)
data["num_features"]=data["features"].apply(len)
data["num_description_words"] = data["description"].apply(lambda x: len(x.split(" ")))

features_to_use  = ["bathrooms", "bedrooms", "price", 
                    "num_photos", "num_features", "num_description_words",                    
                    "created_month", "created_day", "created_hour"
                   ]
                   
train=data[data["Source"]=="train"]
test=data[data["Source"]=="test"]
target_num_map={"high":0, "medium":1, "low":2}
y=np.array(train["interest_level"].apply(lambda x: target_num_map[x]))

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
clf=RFC(n_estimators=1000, random_state=42)
f='llc' #lat long cluster
train[f+'_high']=np.NaN
train[f+'_medium']=np.NaN
train[f+'_low']=np.NaN
k_fold = StratifiedKFold(5, random_state=0)
for (train_index, cv_index) in k_fold.split(train, y):
    clf.fit(train.loc[train.index[train_index], features_to_use], y[train_index])
    y_val_pred = clf.predict_proba(train.loc[train.index[cv_index], features_to_use])
    print(log_loss(y[cv_index], y_val_pred))
    train.loc[train.index[cv_index], [f+'_high', f+'_medium', f+'_low']]=y_val_pred

clf.fit(train[features_to_use], y)
y_test_pred = clf.predict_proba(test[features_to_use])
test[f+'_high']=y_test_pred[:,0]
test[f+'_medium']=y_test_pred[:,1]
test[f+'_low']=y_test_pred[:,2]

train[['listing_id', f+'_high', f+'_medium', f+'_low']].to_csv('train.llc.csv', index=False)
test[['listing_id', f+'_high', f+'_medium', f+'_low']].to_csv('test.llc.csv', index=False)

