import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import random
import time

# random numbers replication
random.seed(1984)

print('Load data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
y_train = train['target']
id_test = test['ID']

print('Clearing...')
# features that are string
types = train.columns.to_series().groupby(train.dtypes).groups
ctext = types[np.dtype('object')]

# fill na with mode for categorical data
for c in ctext:
    mode = train[c].mode()[0]
    train[c] = train[c].fillna(mode)
    mode = test[c].mode()[0]
    test[c] = test[c].fillna(mode)

# transform string columns to numbers
enc = LabelEncoder()
for c in ctext:
    train[c] = enc.fit_transform(train[c])
    test[c] = enc.fit_transform(test[c])

# fill na with mean for numerical data
columns = list(test.columns)
for c in columns:
    mean = train[c].mean()
    train[c] = train[c].fillna(mean)
    mean = test[c].mean()
    test[c] = test[c].fillna(mean)

print('Features selection...')
k = 25  # number of features to select
train = train.drop(['target'], axis=1)
features_names = list(train.columns.values)
train_new = SelectKBest(f_classif, k=k).fit(train, y_train)
fe = train_new.get_support()
features_selected = [features_names[i] for i in list(fe.nonzero()[0])]

# dataset with selected features
train_new = train[features_selected]
test_new = test[features_selected]

print('Training...')
start_time = time.time()
# models parameters
ne = 25  # number of estimators
md = 35  # max depth
g = {'ne': ne, 'md': md, 'mf': k, 'rs': 1984}

clf = ensemble.ExtraTreesClassifier(n_estimators=g['ne'], max_depth=g['md'],
                                    max_features=g['mf'], random_state=g['rs'],
                                    criterion='entropy',
                                    min_samples_split=4,
                                    min_samples_leaf=2, verbose=0, n_jobs=-1)

# GridSearch
y_pred = []
best_score = 0.0
id_results = id_test[:]
LL = make_scorer(log_loss, greater_is_better=False)

model = GridSearchCV(estimator=clf, param_grid={}, n_jobs=-1,
                     cv=2, verbose=0, scoring=LL)
model.fit(train_new, y_train.values)
best_score = (log_loss(y_train.values,
                       model.predict_proba(train_new)))*-1

print('Predict...')
y_pred = model.predict_proba(test_new)[:,1]
print("Ensemble Model: ", c, " Best CV score: ", best_score,
      " Time: ", round(((time.time() - start_time)/60), 2))

for i in range(len(y_pred)):
    if y_pred[i] < 0.0:
        y_pred[i] = 0.0
    if y_pred[i] > 1.0:
        y_pred[i] = 1.0

df_in = pd.DataFrame({"ID": id_test, c: y_pred})
id_results = pd.concat([id_results, df_in[c]], axis=1)
id_results.columns = ['ID', 'PredictedProb']
id_results.to_csv('submission.csv', index=False)
