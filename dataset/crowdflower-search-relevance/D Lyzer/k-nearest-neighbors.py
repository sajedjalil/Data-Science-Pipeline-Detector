"""
	k-Nearest Neighbors
	Crowd Flower - Search Results Relevance
	Author: D Lyzer
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import mode
import scipy.sparse

def similarity( a, b ):
	assert( len( a ) == len( b ) )
	return np.dot(a, b)

def modex( a ):
	return int( mode( a )[0][0] )

# Use Pandas to read in the training and test data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")
train = train.reindex( np.random.permutation( train.index ) )
queries = pd.DataFrame( train['query'] )
uqueries = queries.drop_duplicates()
K = 3 	# K nearest neighbors
y_pred_all = []
id_all = []

for j in range( len( uqueries ) ):
	query_str = uqueries['query'].values[j]
	traind = train[ train['query'] == query_str ]
	testd = test[ test['query'] == query_str ]
	y_train = traind.median_relevance.values
	traindata = list( traind.apply( lambda x:'%s' % ( x['product_title'] ),axis=1 ) )
	testdata = list( testd.apply( lambda x:'%s' % ( x['product_title'] ),axis=1 ) )
	cVecT = TfidfVectorizer( min_df=1,  max_df=1.0, binary=False, max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 5), stop_words = 'english' )
	Z_train = cVecT.fit_transform(traindata).toarray()
	Z_test = cVecT.transform( testdata ).toarray()

	qVec = TfidfVectorizer( min_df=2,  max_df=1.0, binary=False, max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 4), stop_words = 'english' )
	q = traind.iloc[0]['query']
	qVec.fit( [ q, q ] )
	X_pt_q_train = qVec.transform( traindata ).toarray()
	Z_train = np.hstack( [ Z_train, X_pt_q_train ] )
	X_pt_q_test = qVec.transform( testdata ).toarray()
	Z_test = np.hstack( [ Z_test, X_pt_q_test ] )

	y_pred = [ 0 for x in range( len( testdata ) ) ]
	for i in range( len( testdata ) ):
		sim = [ 0 for x in range( len( traindata ) ) ]
		for kn in range( len( traindata ) ):
			sim[kn] = similarity( Z_test[i,:], Z_train[kn,:] )
		traind['sim'] = sim
		kneighbors = traind.sort( ['sim'], ascending=False )[:K]
		y_pred[i] = modex( kneighbors['median_relevance'] )
	y_pred_all = y_pred_all + list( y_pred )
	id_all = id_all + list( testd['id'].values )

submit = pd.DataFrame( {'id': id_all, 'prediction': y_pred_all} )
submit.to_csv( 'knn-cfl.csv', index=False )