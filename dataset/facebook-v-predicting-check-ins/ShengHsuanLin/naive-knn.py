# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KDTree
#from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


############################
#train_accuracy_min=train[['accuracy']].min()
#train_accuracy_max=train[['accuracy']].max()
#train[['accuracy']]=(train[['accuracy']]-train_accuracy_min)/(train_accuracy_max-train_accuracy_min)*10.0
#test[['accuracy']]=(test[['accuracy']]-train_accuracy_min)/(train_accuracy_max-train_accuracy_min)*10.0
############################
#print ("Normalized the accuracy feature")


# Taking the idea from that hour matters
train[['time']]=(train[['time']]/60) % 24
train[['time']]=train[['time']]/24*10
test[['time']]=(test[['time']]/60) % 24
test[['time']]=test[['time']]/24*10


#Weighted y by factor 2
train[['y']]=train[['y']]*2
test[['y']]=test[['y']]*2
#Weighted time by factor 0.5
train[['time']]=train[['time']]/2
test[['time']]=test[['time']]/2



print ("Generated feature hour from time")

tree = KDTree(train[['x', 'y','time']])
_, ind = tree.query(test[['x','y','time']], k=1)
ind1 = [x[0] for x in ind]
test['place_id'] = train.iloc[ind1].place_id.values
test[['row_id', 'place_id']].to_csv('submission_time.gz', index=False, compression='gzip')




# Any results you write to the current directory are saved as output.