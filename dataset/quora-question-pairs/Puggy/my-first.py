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
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords

X_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')

train_qs = pd.Series(X_train['question1'].tolist() + X_train['question2'].tolist()).astype(str)
test_qs = pd.Series(X_test['question1'].tolist() + X_test['question2'].tolist()).astype(str)


words = (" ".join(train_qs)).lower().split()
counts = Counter(words)

stops = set(stopwords.words("english"))

def sentence_match_share(row):
	word_vect = set()
	q1_vect = []
	q2_vect = []
	q1_word = set()
	q2_word = set()
	for word in str(row['question1']).lower().split():
		if word not in stops:
			q1_word.add(word)
	for word in str(row['question2']).lower().split():
		if word not in stops:
			q2_word.add(word)
	word_vect = q1_word | q2_word
	for word in word_vect:
		if word in q1_word:
			q1_vect.append(1)
		else:
			q1_vect.append(0)
		if word in q2_word:
			q2_vect.append(1)
		else:
			q2_vect.append(0)
	x = np.array(q1_vect)
	y = np.array(q2_vect)
	lx = np.sqrt(np.dot(x, x))
	ly = np.sqrt(np.dot(y, y))
	R = np.dot(x, y)/(lx * ly)
	return R

# plt.figure(figsize=(15, 5))
train_word_match = X_train.apply(sentence_match_share, axis=1, raw=True)
# plt.hist(sentence_match_share[X_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
# plt.hist(sentence_match_share[X_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
# plt.legend()
# plt.title('Label distribution over sentence_match_share_share', fontsize=15)
# plt.xlabel('sentence_match_share', fontsize=15)
# plt.show()


x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_test['word_match'] = X_test.apply(sentence_match_share, axis=1, raw=True)

y_train = X_train['is_duplicate'].values

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

from sklearn.cross_validation import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = X_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)
