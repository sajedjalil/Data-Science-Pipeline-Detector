import numpy as np
import pandas as pd

# read data
print( 'read data' )
train_df = pd.read_csv( '../input/train_V2.csv' )
train_df.loc[ 2744604, 'winPlacePerc' ] = 1.0  # It's nan!
train_size = len( train_df.index )
test_df = pd.read_csv( '../input/test_V2.csv' )
test_size = len( test_df.index )

rs = 1
df = pd.concat( [ train_df, test_df ], axis=0, sort=False, ignore_index=True )
del train_df, test_df

train_columns = list( df.columns[ 3:-1 ] )

# one-hot vectorize
print( 'one-hot vectorize' )
m = df[ 'matchType' ]
l = list( set( m.values ) )
d = pd.DataFrame( columns=l, index=df.index )
for t in l:
	i = m[ m==t ].index
	d[ t ].iloc[ i ] = 1
d = d.fillna( 0 )
df = pd.concat( [ df, d ], axis=1 )
del d, m

train_columns.remove( 'matchType' )
train_columns.extend( l )

# manifold
print( 'manifold' )
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

m = { 'pca':PCA( n_components=5, random_state=rs ).fit_transform( df[ train_columns ] ),
	'svd':TruncatedSVD( n_components=5, random_state=rs ).fit_transform( df[ train_columns ] ),
	'ica':FastICA( n_components=5, random_state=rs ).fit_transform( df[ train_columns ] ),
	'grp':GaussianRandomProjection( n_components=5, eps=0.1, random_state=rs ).fit_transform( df[ train_columns ] ),
	'srp':SparseRandomProjection( n_components=5, dense_output=True, random_state=rs ).fit_transform( df[ train_columns ] ) }
manif_columns = [ '%s%d'%( k, a ) for a in range( 5 ) for k in m ]
d = pd.DataFrame( columns=manif_columns, index=df.index, dtype=float )
for key, val in m.items():
	d[ [ '%s%d'%( key, a ) for a in range( 5 ) ] ] = val
df = pd.concat( [ df, d ], axis=1 )
train_columns.extend( list( d.columns ) )
del d, m

# grouping
print( 'grouping' )
d = df[ manif_columns + [ 'groupId' ] ]
g = d.groupby( 'groupId', as_index=False ).mean()
g = d[ manif_columns ]
g.columns = [ 'g_%s'%k for k in g.columns ]
train_columns.extend( list( g.columns ) )
df = pd.concat( [ df, g ], axis=1 )
del d, g

# make match group
print( 'make match group' )
def get_group( df ):
	match_nums = []
	match_ids = []
	match_groups = []
	match_data = np.zeros( ( len( df ), len( train_columns ) ) )
	match_target = np.zeros( ( len( df ), ) )
	pos = 0
	for match in df.groupby( 'matchId' ):
		l = len( match[1] )
		match_nums.append( l )
		match_ids.extend( match[1][ 'Id' ].values )
		match_groups.extend( match[1][ 'groupId' ].values )
		match_data[ pos:pos+l ] = match[1][ train_columns ].values
		match_target[ pos:pos+l ] = match[1][ 'winPlacePerc' ].values
		pos += l
	match_nums = np.array( match_nums )
	return match_ids, match_groups, match_data, match_target, match_nums
_, train_groups, train_data, train_target, train_nums = get_group( df.iloc[ :train_size ] )
valid_ids, valid_groups, valid_data, _, valid_nums = get_group( df.iloc[ train_size: ] )
del df

print( 'GoGoGo!' )
import xgboost as xgb
from xgboost import DMatrix

xgb_params =  {
	'objective': 'rank:pairwise',
	'eta': 0.5,
	'gamma': 0.0001,
	'max_depth': 8,
	'seed': rs,
	'silent': 1
}
xgtrain = DMatrix( train_data, train_target )
xgtrain.set_group( train_nums )

xgb_clf = xgb.train(
	xgb_params,
	xgtrain,
	num_boost_round=100
)
del xgtrain

xgvalid = DMatrix( valid_data )
xgvalid.set_group( valid_nums )
predict_valid = xgb_clf.predict( xgvalid )
del xgvalid, xgb_clf

# rank to score
print( 'rank to score' )
score_p = {}
valid_target = []
p = 0
for n in valid_nums:
	target = predict_valid[ p:p+n ]
	groups = valid_groups[ p:p+n ]
	group_d = dict( zip( groups, target ) )
	group_c = len( group_d )
	if group_c == 1:
		valid_target.extend( [1]*n )
	else:
		rank = sorted( group_d.items(), key=lambda x: x[1] )
		rank_g = [ r[0] for r in rank ]
		for g in groups:
			idx = rank_g.index( g )
			score = idx / ( group_c - 1 )
			valid_target.append( score )
	p += n

result = pd.DataFrame( { 'Id':valid_ids, 'winPlacePerc':valid_target } )
result.to_csv( 'result.csv', index=None )

# finish
print( 'finish' )
