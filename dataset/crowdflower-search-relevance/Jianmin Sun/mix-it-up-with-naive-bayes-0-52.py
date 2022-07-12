"""
Mix it up with Naive Bayes - 0.52+
Crowdflower's Search Results Relevance @ Kaggle
__author__ : Darragh
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB

# Load the training and test file
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Drop the ID columns
idx = test.id.values.astype(int)
train, test = train.drop('id', axis=1), test.drop('id', axis=1)

# create labels and drop variance
y = train.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

# join query to one set, and query/title to another
trainjoin1 = list(train.apply(lambda x:'%s' % (x['query']),axis=1))
trainjoin2 = list(train.apply(lambda x:'%s %s' % (x['product_title'], x['query']),axis=1))
testjoin1 = list(test.apply(lambda x:'%s' % (x['query']),axis=1))
testjoin2 = list(test.apply(lambda x:'%s %s' % (x['product_title'], x['query']),axis=1))

# Create tfidf vectoriser
tfv = TfidfVectorizer(min_df=2,  max_features=None, strip_accents='unicode', 
        analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 5), use_idf=1,
        smooth_idf=1,sublinear_tf=1, stop_words = 'english')

# Fit TFIDF on train and test
tfv.fit(trainjoin1)
X1, X1_test =  tfv.transform(trainjoin1), tfv.transform(testjoin1)
tfv.fit(trainjoin2)
X2, X2_test =  tfv.transform(trainjoin2), tfv.transform(testjoin2)
X, X_test = hstack([X1, X2]), hstack([X1_test, X2_test]) 

# Fit Naive Bayes Model
nbmodel = MultinomialNB(alpha=.0003)
nbmodel.fit(X, y)
preds = nbmodel.predict(X_test)

# Create your naive bayes submission file
submission = pd.DataFrame({"id": idx, "prediction": preds})
submission.to_csv("naive_bayes.csv", index=False)