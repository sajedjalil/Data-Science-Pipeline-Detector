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
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

np.random.seed(69)

train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

log_reg = LogisticRegression(C=2000, multi_class="multinomial", tol=0.0001, solver='newton-cg')
log_reg.fit(x_train, y_train)
model = SelectFromModel(log_reg, prefit=True)
x_train_new = model.transform(x_train)
x_test_new = model.transform(x_test)
log_reg.fit(x_train_new, y_train)
y_test = log_reg.predict_proba(x_test_new)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('output_logistic.csv')