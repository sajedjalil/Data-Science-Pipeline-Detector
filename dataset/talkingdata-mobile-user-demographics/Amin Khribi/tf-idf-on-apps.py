"""
In this script, I compose tf-idf trasformation of apps (installed and active \
all together) in a sparse matrix, then perform a cross-validated logistic regression to \
classify devices. 
"""

import pandas as pd
import numpy as np

import random

from scipy import sparse

from sklearn import metrics, linear_model, cross_validation, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from gensim import models

import re

from nltk.corpus import stopwords


######################
##### LOADING ########
######################

print("loading csv ...")

df_events = pd.read_csv('../input/events.csv', dtype={'event_id': 'str', 'device_id': 'str'})
df_app_events = pd.read_csv('../input/app_events.csv', dtype='str')

df_app_labels = pd.read_csv('../input/app_labels.csv', dtype='str')
df_label_categories = pd.read_csv('../input/label_categories.csv', dtype='str')

df_gender_age_train = pd.read_csv('../input/gender_age_train.csv', dtype='str')


##########################
####### MAPPING ##########
##########################

#
print("map device_id")
labels = sorted(set(df_gender_age_train['device_id'].unique()).\
				union(df_events['device_id'].unique()))

le_device = LabelEncoder()
le_device.fit(labels)

df_events['device_id']= le_device.transform(df_events['device_id'].values)
df_gender_age_train['device_id']= le_device.transform(df_gender_age_train['device_id'].values)

#
print("map app_id")
labels = sorted(set(df_app_labels['app_id'].unique()).union(df_app_events['app_id'].unique()))

le_app = LabelEncoder()
le_app.fit(labels)

df_app_events['app_id'] = le_app.transform(df_app_events['app_id'].values)
df_app_labels['app_id'] = le_app.transform(df_app_labels['app_id'].values)

#
print("map group")

le_group = LabelEncoder()
df_gender_age_train['group'] = le_group.fit_transform(df_gender_age_train['group'].values)


####
#

Y_all = df_gender_age_train["group"].values

Y_gender = df_gender_age_train["gender"].values
Y_age = df_gender_age_train["age"].astype('float').values

all_index = list(range(len(Y_all)))
random.shuffle(all_index)

idx_learn = list(all_index[:int(0.75 * len(all_index))])
idx_test = list(set(all_index).difference(idx_learn))

all_devices = df_gender_age_train['device_id'].values

uniques_learn = [all_devices[k] for k in idx_learn]
uniques_test = [all_devices[k] for k in idx_test]




#############################
###### TF IDF ON APPS #######
#############################
print('tfidf on apps')

df_app_events = df_app_events.merge(df_events[['event_id', 'device_id']], on='event_id' ,how='right')

all_devices = df_gender_age_train['device_id'].values

train_tr = df_gender_age_train.merge(df_app_events[['device_id', 'app_id']], on='device_id', how='left')


# compose  tfidf matrix

dict_cor = {v: k for k, v in enumerate(all_devices)}
dict_cor_inv = {v: k for k, v in dict_cor.items()}

corpus = []
indexes = []

groups = train_tr[['device_id', 'app_id']].groupby('device_id')  # preservs order of device_id

# train
for name, x in groups:

	to_add = list(x['app_id'].value_counts().to_dict().items())

	if len(to_add):
		corpus.append(to_add)
		indexes.append(name)

tfidf = models.TfidfModel(corpus)


n_devices = train_tr['device_id'].nunique()
n_feature = train_tr['app_id'].nunique()

data = []
col = []
row = []

# fill rows according to all_devices order

for i in train_tr['device_id'].unique():

	if i in indexes:
		corps = tfidf[corpus[indexes.index(i)]]

		row.extend([dict_cor[i]] * len(corps))
		col.extend([k[0] for k in corps])
		data.extend([k[1] for k in corps])

col = LabelEncoder().fit_transform(col)

X_app_tfidf = sparse.csr_matrix((data, (row, col)), shape=(n_devices, n_feature))



#########################
####### LEARNING ########
#########################
print('logistic regression')


X_learn, Y_learn = X_app_tfidf[idx_learn, :], Y_all[idx_learn]
X_test, Y_test = X_app_tfidf[idx_test, :], Y_all[idx_test]

###
dict_cv={"C": np.linspace(0.01, 15, 7), "penalty": ['l1', 'l2']}

gs = grid_search.GridSearchCV(linear_model.LogisticRegression(),
                              dict_cv,
                              scoring='log_loss',
                              n_jobs=-1,
                              iid=False,
                              refit=True,
                              cv=cross_validation.StratifiedKFold(Y_learn),
                              verbose=0)

gs.fit(X_learn, Y_learn)

bp = gs.best_params_
print(bp)
clf = gs.best_estimator_

y_hat = clf.predict_proba(X_test)


score = metrics.log_loss(Y_test, y_hat)