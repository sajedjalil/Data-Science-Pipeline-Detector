
import pandas as pd
import numpy as np
from itertools import islice
import xgboost as xgb
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from numpy.core.multiarray import ndarray
from sklearn.feature_extraction import text
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2,SelectPercentile


from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import brown
from datetime import date, time

from datetime import timedelta
from pandas.tseries.tools import to_datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score

from sklearn.preprocessing import LabelEncoder
from collections import Counter,defaultdict
import math
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.decomposition import TruncatedSVD
import nltk
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics, ensemble, linear_model, svm
from numpy import log, ones, array, zeros, mean, std



import scipy.sparse as sp
from time import time
from sklearn.cross_validation import StratifiedKFold,train_test_split,KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from nltk.stem import LancasterStemmer,SnowballStemmer,PorterStemmer
import re
from collections import Counter
import string
from nltk.corpus import brown
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import OrderedDict
import itertools
from sklearn import decomposition, pipeline, metrics, grid_search
import collections

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

# set some nicer defaults for matplotlib
from matplotlib import rcParams
from nltk import word_tokenize


rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
#rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def gini(solution, submission):                                                 
    df = sorted(zip(solution, submission),    
            key=lambda x: x[1], reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]                
    totalPos = np.sum([x[0] for x in df])                                       
    cumPosFound = np.cumsum([x[0] for x in df])                                     
    Lorentz = [float(x)/totalPos for x in cumPosFound]                          
    Gini = [l - r for l, r in zip(Lorentz, random)]                             
    return np.sum(Gini)                                                         


def normalized_gini(solution, submission):                                      
    normalized_gini = gini(solution, submission)/gini(solution, solution)       
    return normalized_gini   

train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)
labels = train.Hazard

columns = train.columns
test_ind = test.ix[:,'Id']

train.drop('Hazard', axis=1, inplace=True)
train.drop('Id',axis=1,inplace=True)
test.drop('Id', axis=1, inplace=True)


train = np.array(train)
test = np.array(test)

for i in range(train.shape[1]):
    lbl = LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])
    
train = train.astype(float)
test = test.astype(float)

est = pipeline.Pipeline([
                        ('model', xgb.XGBRegressor())
                        ])
param_grid =    { 'model__min_child_weight': [5],
                  'model__colsample_bytree': [0.8],
                  'model__subsample':[0.8],
                  'model__max_depth': [5,7],
                  'model__learning_rate': [0.01,0.005,0.001],
                  'model__n_estimators': [1000,3000]
                  }

gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

# Initialize Grid Search Modelg
model = grid_search.GridSearchCV(estimator  = est,param_grid = param_grid,scoring = gini_scorer,verbose= 1,iid= True,
                                     refit = True,cv  = 3)
model.fit(train, labels)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


