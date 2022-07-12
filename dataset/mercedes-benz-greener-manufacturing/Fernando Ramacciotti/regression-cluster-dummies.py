# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

X = train.iloc[:, 2:]
y = train.iloc[:, 1]
test = test_data.iloc[:, 1:]

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
for i in range(0, 8):
    X.iloc[:, i] = labelencoder_X.fit_transform(X.iloc[:, i])
    test.iloc[:, i] = labelencoder_X.fit_transform(test.iloc[:, i])
    
"""
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, 'ward')

from scipy.cluster.hierarchy import fcluster
max_d = 25
clusters = fcluster(Z, max_d, criterion = 'distance')
#clusters_test = fcluster(Z_test, max_d, criterion = 'distance')
print(max(clusters))
"""
#Z_test = linkage(test, 'ward')

from sklearn.cluster import KMeans
cluster = KMeans(n_clusters = 210, init = 'k-means++', random_state = 42, n_init = 30)
cluster.fit(X, y)

cluster_dummies = pd.get_dummies(cluster.predict(X))

cluster_dummies_test = pd.get_dummies(cluster.predict(test))
#print(cluster_dummies)

#X = pd.concat([X, cluster_dummies], axis = 1, join = 'inner', ignore_index = True)
#test = pd.concat([test, cluster_dummies_test], axis = 1, join = 'inner', ignore_index = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
cl_d_train, cl_d_test, y_train, y_test = train_test_split(cluster_dummies, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(cl_d_train, y_train)

y_pred = regressor.predict(cl_d_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_true = y_test, y_pred = y_pred)
print(r2)


y_pred_test = regressor.predict(cluster_dummies_test)
output = pd.DataFrame()
output['ID'] = test_data['ID']
output['y'] = y_pred_test
output.to_csv('cluster_dummy.csv', sep = ',', index = False)
