# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.preprocessing as prep
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data.info()
objects = [i for i in data.columns if data[i].dtype == 'object']
ndata = data.drop(objects, 1)
ndata.dropna(inplace=True)

mask = np.random.random(len(ndata)) < 0.8
train, validate = ndata[mask], ndata[~mask]
X, Y = train.drop(['ID', 'target'], axis=1), train.target

cls = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=15,
                                          n_jobs=-1, criterion='gini')
cls.fit(X, Y)

_X = validate.drop(['ID', 'target'], axis=1)
result = cls.predict_proba(_X)
print(metrics.log_loss(validate.target, result[:, 1]))

ntest = test.drop(objects, axis=1)
ntest = ntest.drop('ID', axis=1)

X = prep.Imputer().fit_transform(ntest)
results = cls.predict_proba(X)
results
sub = pd.DataFrame({'ID': test.ID,
                    'PredictedProb': results[:,1]})
sub[['ID', 'PredictedProb']].to_csv('submission.csv', index=False)