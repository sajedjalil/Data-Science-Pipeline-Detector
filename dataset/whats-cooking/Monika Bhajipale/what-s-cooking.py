import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle as pickle
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile, f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from helper import *

#############
### Grids ###
#############

ingred_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(strip_accents='unicode',analyzer="char",preprocessor=stripString)),
    ('feat', SelectPercentile(chi2)),
    ('model', LogisticRegression())
])
ingred_grid = {
    'tfidf__ngram_range':[(2,6)],
    'feat__percentile':[95,90,85],
    'model__C':[5]
}

pipe_glm = Pipeline([
    ('tfidf', TfidfVectorizer(strip_accents='unicode',
    	analyzer="char",preprocessor=stripString)),
    ('feat', SelectPercentile(chi2)),
    ('model', LogisticRegression())
])
grid_glm = {
	'greek':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [37]
	},
	'southern_us':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [57]
	},
	'filipino':{
		'model__C': [7], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [65]
	},
	'indian':{
		'model__C': [3], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [52]
	},
	'jamaican':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [99]
	},
	'spanish':{
		'model__C': [3], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [88]
	},
	'italian':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [91]
	},
	'mexican':{
		'model__C': [7], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [93]
	},
	'chinese':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [76]
	},
	'british':{
		'model__C': [10], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [75]
	},
	'thai':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [97]
	},
	'vietnamese':{
		'model__C': [3], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [27]
	},
	'cajun_creole':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [88]
	},
	'brazilian':{
		'model__C': [10], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [98]
	},
	'french':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [46]
	},
	'japanese':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [97]
	},
	'irish':{
		'model__C': [8], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [17,15,13]
	},
	'korean':{
		'model__C': [8], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [54]
	},
	'moroccan':{
		'model__C': [10], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [55]
	},
	'russian':{
		'model__C': [6], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [91]
	}
}

