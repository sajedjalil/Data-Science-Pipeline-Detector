# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as ssp

from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

tfidf = TfidfVectorizer(
	min_df=5, max_features=500, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(train_text_df["Text"])

test_data = train_text_df.append(test_text_df)
X_tfidf_text = tfidf.transform(test_data["Text"])

#Feature reduction. 
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(200)
SVD_data = svd.fit_transform(X_tfidf_text)

X_train_text = SVD_data [:train_text_df.shape[0]]
X_test_text = SVD_data [train_text_df.shape[0]:]

features = tfidf.get_feature_names()

ID_train = train_variants_df.ID
ID_test = test_variants_df.ID

y = train_variants_df.Class.values-1

train_variants_df = train_variants_df.drop(['ID','Class'], axis=1)
test_variants_df = test_variants_df.drop(['ID'], axis=1)

data = train_variants_df.append(test_variants_df)

X_data = pd.get_dummies(data).values

X = X_data[:train_variants_df.shape[0]]
X_test = X_data[train_variants_df.shape[0]:]

X = ssp.hstack([pd.DataFrame(X_train_text), X], format='csr')
X_test = ssp.hstack((pd.DataFrame(X_test_text), X_test), format='csr')

y_test = np.zeros((X_test.shape[0], max(y)+1))

#Try different classifiers
import xgboost as xgb
clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
clf.fit(X, y)

t_test = clf.predict_proba(X_test)

classes = "class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
subm = pd.DataFrame(t_test, columns=classes)
subm['ID'] = ID_test

subm.to_csv('submission.csv', index=False)	