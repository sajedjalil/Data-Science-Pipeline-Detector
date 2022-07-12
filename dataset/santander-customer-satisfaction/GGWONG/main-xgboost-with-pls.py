import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster
from sklearn import ensemble
from sklearn import cross_validation

from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFECV, SelectFromModel
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score as auc
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

import xgboost as xgb

from operator import itemgetter
from scipy.stats import randint
from scipy.stats import uniform
import time

plt.rcParams['figure.figsize'] = (10, 10) #plt style sheet 
