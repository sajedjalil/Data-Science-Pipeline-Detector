# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')



#make a copy of the training dataset


train_copy = train.copy()

#Lets drop the groupID and matchID

train_noid=train_copy.drop("matchId",axis=1)
train_noid1 = train_noid.drop("groupId",axis=1)

#Train_noid1 has no matchID or GroupID variable. 

#Find some Correlations
corr_matrix = train_noid1.corr()
#Sort by corr
corr_matrix['winPlacePerc'].sort_values(ascending=False)


#PCA exploration
train_prepared = train_noid1.drop("winPlacePerc",axis=1)
train_labels = train_noid1["winPlacePerc"].copy()

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
train_reduced = pca.fit_transform(train_prepared)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train_reduced, train_labels)
forest_predictions = forest_reg.predict(train_reduced)


from sklearn.metrics import mean_absolute_error
forest_mae = mean_absolute_error(train_labels, forest_predictions)

#ok it sucks but lets predict on the test set:
test_copy = test.copy()
test_noid = test_copy.drop("matchId",axis=1)
test_noid1 = test_noid.drop("groupId",axis=1)
test_prepared = test_noid1.copy()
pca = PCA(n_components=0.95)
test_reduced = pca.fit_transform(test_prepared)
test_predictions = forest_reg.predict(test_reduced)

test_submission = test.copy()
test_submission["winPlacePerc"] = test_predictions
submission = test_submission[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)
#moving forward:
