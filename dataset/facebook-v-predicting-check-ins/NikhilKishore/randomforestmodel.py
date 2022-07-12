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
## Reading Data
test= pd.read_csv("../input/test.csv")
train= pd.read_csv("../input/train.csv")
## Taking a small subsection of the data

a= train[(train.x>0) & (train.x<0.25) & (train.y>0) & (train.y<0.25)] 
## Subsetting time 
a['hours'] = a.time/60 % 24
a['weekday'] = (a.time/(60*24)) % 7
a['month'] = (a.time/(60*24*30)) % 12
a['year'] = a.time/(60*24*365)
a['day'] = a.time/(60*24) % 365
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
jobs=multiprocessing.cpu_count()
from sklearn.metrics import accuracy_score
## there are approximately 550 unique places in the block therefore doing some feature engineering
## by adding Kmeans clusters as a feature 
from sklearn.cluster import KMeans
model=KMeans(n_clusters=550).fit_predict(a)
a["clusters"] = model
a.head()

## Subsetting data to train and validation set
trainsmall,valid = train_test_split(a, test_size=0.33, random_state=True)
y=trainsmall.place_id
x=trainsmall[['x','y','accuracy','time','hours','weekday','clusters','month','year','day']]
pred_valid=valid[['x','y','accuracy','time','hours','weekday','clusters','month','year','day']]
label_valid= valid.place_id
jobs=multiprocessing.cpu_count()
#Running Random Forest Classifier  
rf=RandomForestClassifier(n_estimators=100, n_jobs=jobs, max_depth=18,min_samples_leaf=1,
                            min_samples_split=1, verbose=0, oob_score = True)
rf.fit(X=x, y=y)

trainpredictionsValid = rf.predict(pred_valid)
valid0 = valid.copy()
valid0['pred'] = trainpredictionsValid
ac=accuracy_score(valid0.pred,valid0.place_id)
 
print ("Accuracy is {0}".format(ac))

