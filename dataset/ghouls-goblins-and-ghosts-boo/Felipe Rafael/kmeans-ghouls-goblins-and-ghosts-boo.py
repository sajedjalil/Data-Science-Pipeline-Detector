# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Save the ids and remove to process
test_id = test['id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

#Classes for numerical
train['color'] = train['color'].map({'white':0,'black':1,'clear':2,'blue':3,'green':4,'blood':5}).astype(float)
test['color'] = test['color'].map({'white':0,'black':1,'clear':2,'blue':3,'green':4,'blood':5}).astype(float)


from sklearn.preprocessing import LabelEncoder
#Separates data and type
X_train = train.drop('type', axis=1)
lab_enc = LabelEncoder()
Y_train = lab_enc.fit_transform(train.type.values)




#print(X_train)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
kmeans_pred =  kmeans.predict(test)

kmeans_pred = lab_enc.inverse_transform(kmeans_pred)
#print(kmeans_pred)

output = pd.DataFrame({'id':test_id, 'type':kmeans_pred})
output.to_csv('output_submission.csv', index=False)



