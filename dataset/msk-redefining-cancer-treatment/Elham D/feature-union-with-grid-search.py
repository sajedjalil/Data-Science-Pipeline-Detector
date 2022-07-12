# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Fri Jul 28 11:04:20 2017
#
#@author: elhamdolatabadi
#"""
#
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import seaborn as sns
#from nltk.tokenize import RegexpTokenizer
#from nltk.stem.porter import PorterStemmer
#from stop_words import get_stop_words
from __future__ import print_function
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn import model_selection

from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, RandomizedSearchCV, GridSearchCV


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.metrics import classification_report, roc_curve ,auc
from scipy.stats import randint, expon

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.model_selection import StratifiedKFold        
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder


#%matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/test_variants')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 
            
            
train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]            

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

print('Pipeline...')
print('Pipeline...')
fp = Pipeline([
    ('union', FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', Pipeline([('Gene', cust_txt_col('Gene')), ('vect_Gene', CountVectorizer(analyzer=u'char',max_df = 0.5,ngram_range=(1, 2),max_features=None)), ('tsvd1', TruncatedSVD(n_iter=25, random_state=12,n_components=10))])),
           # ('pi2', Pipeline([('Variation', cust_txt_col('Variation')), ('vect_Variation', CountVectorizer(analyzer=u'char',max_features=None,max_df = 0.5,ngram_range=(1, 2))), ('tsvd2', TruncatedSVD(n_iter=25, random_state=12,n_components=10))])),
            #commented for Kaggle Limits
           # ('pi3', Pipeline([('Text', cust_txt_col('Text')), ('tfidf', TfidfVectorizer()), ('tsvd3', TruncatedSVD(n_iter=25, random_state=12,n_components=10))]))
        ])
    )])
  

parameters = {
    'union__pi1__vect_Gene__max_df': (0.5, 0.75, 1.0),
 #   'union__pi1__vect_Gene__max_features': (None, 5000, 10000, 50000),
 #   'union__pi1__vect_Gene__ngram_range': ((1, 2),(1,3),(1,4),(1,5))  # unigrams or bigrams'vect_Gene__max_df': (0.5, 0.75, 1.0),
 #   'union__pi2__vect_Variation__max_features': (None, 5000, 10000, 50000),
 #   'union__pi2__vect_Variation__max_df': (0.5, 0.75, 1.0),
 #    'union__pi2__vect_Variation__ngram_range': ((1, 2),(1,3),(1,4),(1,5),(1,8)),  # unigrams or bigrams
 #   'union__pi1__tsvd1__n_components' : (10,20,100)
 #    'union__pi2__tsvd2__n_components' : (10,20,30),
  #  'union__pi3__tsvd3__n_components' : (5,10,100),
#    'union__pi3__tfidf__use_idf': (True, False),
#    'union__pi3__tfidf__norm': ('l1', 'l2'),
#    'union__pi3__tfidf__ngram_range': ((1, 2),(1,8)),
#    'clf__alpha': (0.00001, 0.000001),
#    'clf__penalty': ('l2', 'elasticnet'),
#    'clf__n_iter': (10, 50, 80),
}
    
y = y - 1 #fix for zero bound array
if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(fp, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in fp.steps])
    print("parameters:")
    pprint(parameters)
    grid_search.fit(train,y)
    train = grid_search.transform(train)
    print(train.shape)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    




# Any results you write to the current directory are saved as output.