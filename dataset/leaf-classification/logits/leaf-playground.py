import time
import statistics
import collections
import numpy
from IPython.display import display, FileLinks
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

start = time.time()

train = pd.read_csv('../input/train.csv')
x = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'].values)
scaler = StandardScaler().fit(x)
x = scaler.transform(x)
y = le.transform(train['species'])

c = LogisticRegression(C=3000)#, multi_class='multinomial', solver='sag')

accs = []
losses = []
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=23)
for train_index, test_index in sss.split(x, y):
    c.fit(x[train_index], y[train_index])
    acc = accuracy_score(y[test_index], c.predict(x[test_index]))
    loss = log_loss(y[test_index], c.predict_proba(x[test_index]))
    display('{} {}'.format(acc, loss))
    accs.append(acc)
    losses.append(loss)

display('---')
display('{} {}'.format(statistics.mean(accs), statistics.mean(losses)))
display('it took {}s'.format(time.time() - start))


c = c.fit(x, y)

test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
x_test = scaler.transform(x_test)

y_test = c.predict_proba(x_test)
pd.DataFrame(y_test, index=test_ids, columns=le.classes_).to_csv('result.csv')
