# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")
sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
test = pd.read_csv("../input/Kannada-MNIST/test.csv")
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

from sklearn.ensemble import RandomForestClassifier

train_data = train.iloc[0:60000,1:]
train_label = train.iloc[0:60000,0]

clf = RandomForestClassifier(n_estimators =20)
clf.fit(train_data,train_label)

test_data = test.iloc[0:5000,1:]
#print(test_data)
predict = clf.predict(test_data)

test_id = test.iloc[0:5000,0]
#print(test_id)

Kagglesubmission = pd.DataFrame({'id': test_id, 'label': predict})
Kagglesubmission.to_csv('submission.csv', index=False)