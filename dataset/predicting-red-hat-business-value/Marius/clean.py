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

train=pd.read_csv('../input/act_train.csv')
test=pd.read_csv('../input/act_test.csv')
sub=pd.read_csv('../input/sample_submission.csv')

remove_type=['activity_category', 'char_1',
       'char_2', 'char_3', 'char_4', 'char_5', 'char_6', 'char_7', 'char_8',
       'char_9', 'char_10']
train[remove_type]=train[remove_type].fillna('0')
test[remove_type]=test[remove_type].fillna('0')
for i in remove_type:
	train[i]=train[i].str.replace('type ','')
	test[i]=test[i].str.replace('type ','')

# people_unique=train['people_id'].unique()

# for u in people_unique:
#     train[train['people_id'] == u]['act_time']=train[train['people_id'] == u]['people_id'].count()

	
# train.to_csv('train_clean.csv')

key_use=['activity_category','char_1','char_10']

from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn import ensemble

# model=RFE(clf,3)
# model.fit(train[remove_type],train['outcome'])
# # j=model.predit(test[remove_type])
# print(model.support_)
# print(model.ranking_)

clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=1)
clf = clf.fit(train[remove_type], train['outcome'])
j=clf.predict(test[remove_type])
print(j)
sub['outcome']=j
print(sub)
sub.to_csv('submission.csv',index=False)
