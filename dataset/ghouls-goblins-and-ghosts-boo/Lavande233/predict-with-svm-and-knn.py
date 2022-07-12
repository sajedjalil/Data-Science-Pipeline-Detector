# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import scipy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC


# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score,
                                                                     np.std(score.cv_validation_scores)),
              "Parameters: {0}".format(score.parameters))
    return

param_dist = {'C': scipy.stats.expon(scale=100), 
              'gamma': scipy.stats.expon(scale=.1),
              'kernel': ['rbf'],
              'class_weight':['balanced', None]}
clf = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=20)
clf = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type'])
grid_scores = clf.grid_scores_
report(grid_scores=grid_scores)
predict1 = clf.predict(test[['bone_length','rotting_flesh','hair_length']])

param_dist = {"n_neighbors":sp_randint(10,20),
             'weights':('uniform','distance')}

clf = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=param_dist, n_iter=20)
clf = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type'])
grid_scores = clf.grid_scores_
report(grid_scores=grid_scores)
predict2 = clf.predict(test[['bone_length','rotting_flesh','hair_length']])

#Mean validation score: 0.704 (std: 0.016) Parameters: {'weights': 'distance', 'n_neighbors': 15}

param_dist = {"max_depth": sp_randint(1,11),
              "min_samples_split": sp_randint(2, 11),
              "subsample": [0.2, 0.5],
              "min_samples_leaf": sp_randint(5, 11),
              "learning_rate": [0.1, 0.01]}

clf = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=param_dist, n_iter=20)
clf = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type'])
grid_scores = clf.grid_scores_
report(grid_scores=grid_scores)
predict3 = clf.predict(test[['bone_length','rotting_flesh','hair_length']])


param_dist = {'n_estimators':[10,20,50],
              'bootstrap':[True,False],
              'criterion':['gini','entropy'],
              "max_depth": sp_randint(1,11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11)}

clf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=40)
clf = clf.fit(train[['id','has_soul','bone_length','rotting_flesh','hair_length']],train['type'])
grid_scores = clf.grid_scores_
report(grid_scores=grid_scores)
predict4 = clf.predict(test[['id','has_soul','bone_length','rotting_flesh','hair_length']])

param_dist = {'n_estimators':[10,20,50,100],
              'bootstrap':[True,False],
              'criterion':['gini','entropy'],
              "max_depth": sp_randint(1,11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11)}

clf = RandomizedSearchCV(ExtraTreesClassifier(), param_distributions=param_dist, n_iter=40)
clf = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type'])
grid_scores = clf.grid_scores_
report(grid_scores=grid_scores)
predict5 = clf.predict(test[['bone_length','rotting_flesh','hair_length']])

pred = {'id':test['id'],
        'svm':predict1,
        'knn':predict2,
        'Gradient boosting':predict3,
        'random forest':predict4,
        'extra tree':predict5}
pred = pd.DataFrame(pred)

submission = pred = {'id':test['id'],
        'type':predict4}
submission = pd.DataFrame(submission)
submission.to_csv('random_forest.csv', index=False)