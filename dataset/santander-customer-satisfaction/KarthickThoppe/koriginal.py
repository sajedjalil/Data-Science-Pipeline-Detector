
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
import numpy as np
import pandas as pd
import re

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
#%matplotlib inline

pd.set_option('display.precision', 5)

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET



y.value_counts() / float(y.size)

# ratio of nonzero elements
X.apply(lambda x:x[x!=0].size).sum() / float(np.prod(training.shape))

test.apply(lambda x:x[x!=0].size).sum() / float(np.prod(test.shape))

X.dtypes.value_counts()

X.columns

name_component = pd.Series(sum([re.sub("\d+", "", s).split("_") for s in X.columns], []))
name_component.replace("", "_0", inplace=True)
name_component.value_counts()

nuniques_train = X.apply(lambda x:x.nunique())
nuniques_test = test.apply(lambda x:x.nunique())

no_variation_train = nuniques_train[nuniques_train==1].index
no_variation_test = nuniques_train[nuniques_test==1].index

print(no_variation_train.size, no_variation_test.size)

print('\nTrain[no variation in test]\n#unique cnt\n',nuniques_train[no_variation_test].value_counts())
print('\nTest[no variation in train]\n#unique cnt\n', nuniques_test[no_variation_train].value_counts())


X, test = [df.drop(no_variation_train, axis=1) for df in [X, test]]
nuniques_train, nuniques_test = [s.drop(no_variation_train) for s in [nuniques_train, nuniques_test]]

ax = nuniques_train[nuniques_train<100].hist(bins=100, figsize=(10, 7))
ax.set_xlabel("#uniques")
ax.set_title("Histogram of #uniques (<100)")
plt.show()
nuniques_train[nuniques_train<100].size

nuniques_train[nuniques_train>=100].size

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values(inplace=True)
ax = feat_imp.tail(20).plot(kind='barh', figsize=(10,7), title='Feature importance')

