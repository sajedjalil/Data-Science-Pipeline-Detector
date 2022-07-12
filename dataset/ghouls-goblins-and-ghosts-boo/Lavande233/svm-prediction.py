# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
# Input data files are available in the "../input/" directory.
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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score,
                                                                     np.std(score.cv_validation_scores)),
              "Parameters: {0}".format(score.parameters))
    return

# Any results you write to the current directory are saved as output.
param_dist = {'C': scipy.stats.expon(scale=100), 
              'gamma': scipy.stats.expon(scale=.1),
              'kernel': ['rbf'],
              'class_weight':['balanced', None]}
clf = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=40)
clf = clf.fit(train[['has_soul','bone_length','rotting_flesh','hair_length']],train['type'])
grid_scores =clf.grid_scores_
report(grid_scores=grid_scores)
pred = clf.predict(test[['has_soul','bone_length','rotting_flesh','hair_length']])

submission = pred = {'id':test['id'],
        'type':pred}
submission = pd.DataFrame(submission)
submission.to_csv('svm.csv', index=False)