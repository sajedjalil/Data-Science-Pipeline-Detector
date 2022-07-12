import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")


train_data = train.drop('Id', axis=1)
test_data = test.drop('Id', axis=1)

train_labels = train_data['Cover_Type']
train_data = train_data.drop('Cover_Type',axis=1)

# print(test_data.shape)
# print(train_data.shape)

clf = KNeighborsClassifier()
clf.fit(train_data, train_labels)
test_data['Cover_Type'] = pd.Series(clf.predict(test_data))

submission = pd.concat([test['Id'].astype(int), test_data['Cover_Type']], axis=1)
submission = submission.set_index('Id')
submission.to_csv('submission.csv')

print(submission.shape)




