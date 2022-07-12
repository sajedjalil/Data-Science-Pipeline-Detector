
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# we dont need ID columns
idx = test.id.values.astype(int)
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# create labels. drop useless columns
y = train.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

# do some lambda magic on text columns
traindata = list(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
testdata = list(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))

# the infamous tfidf vectorizer (Do you remember this one?)
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

# Fit TFIDF
tfv.fit(traindata)
X =  tfv.transform(traindata) 
X_test = tfv.transform(testdata)

# LSA / SVD
svd = TruncatedSVD(n_components = 200)
X = svd.fit_transform(X)
X_test = svd.transform(X_test)

# Scaling the data is important prior to SVM
scl = StandardScaler()
X = scl.fit_transform(X)
X_test = scl.transform(X_test)

#model = SVC(C=10.0)
model = GradientBoostingRegressor(n_estimators=50,max_depth=6,learning_rate=0.1, min_samples_split=1,verbose=1)

# Fit SVM Model
model.fit(X, y)
preds = model.predict(X_test)
#print(preds[1:5])
#print(type(preds))

# Create your first submission file
submission = pd.DataFrame({"id": idx, "prediction": preds.astype(int)})
print(submission[1:2,])
submission.to_csv("abhishek_bigrams_gbm_100-6-01.csv", index=False)
