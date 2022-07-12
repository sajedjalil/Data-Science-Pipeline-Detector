# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# print(train[:10])

pivot = int(0.8 * len(train))
feature_labels = train.columns.values.tolist()[:-1] # remove the TARGET column

features = np.array(train[feature_labels])
target = np.array(train['TARGET'])

validation_features = features[pivot:]
train_features = features[:pivot]

validation_target = target[pivot:]
train_target = target[:pivot]

print("Len validation:", len(validation_features), "Len train:", len(train_features))

model = LogisticRegression()

model.fit(train_features, train_target)
score = model.score(validation_features, validation_target)

print(score)

