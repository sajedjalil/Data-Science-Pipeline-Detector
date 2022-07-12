# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# ----------- 

df_train = pd.read_csv( "../input/clicks_train.csv", nrows=100000)

# ----------- 
# model training goes here ... let's borrow from the "pandas is cool" script as a quick example
# ----------- 

ad_likelihood = df_train.groupby('ad_id').clicked.agg(['count','sum' ]).reset_index()
M             = df_train.clicked.mean()

ad_likelihood['likelihood'] = (ad_likelihood['sum'] + 12*M) / (12 + ad_likelihood['count'])

df_train = df_train.merge(ad_likelihood, how='left')
df_train.likelihood.fillna(M, inplace=True)

# ----------- 
# NOTE: this approach is specific to this particular competition
#
# The MAP metric here just boils down to knowing where you ended up ranking the actual ad that was clicked, relative
# to the other ads in its display context.
# ----------- 

df_train.sort_values(['display_id', 'likelihood'], inplace=True, ascending=[True, False] )

# -------
# Slower way
#

from ml_metrics import mapk

Y_ads = df_train[ df_train.clicked == 1 ].ad_id.values.reshape(-1,1)
P_ads = df_train.groupby(by='display_id', sort=False).ad_id.apply( lambda x: x.values ).values

score = mapk( Y_ads, P_ads, 12 )

print("MAP: %.12f" % score)

# -------
# Now this is a quicker way to evaluate your score without needing to groupby or use the default MAP@ functions.
#
# After sorting each context in order of decreasing predicted probability, and giving each row in the dataset a sequential 
# index, then the delta between the index of the clicked ad, and the index of the first ad in the context, will give you
# the relative rank of the clicked ad within each context.

df_train["seq"] = np.arange(df_train.shape[0])
Y_seq           = df_train[ df_train.clicked == 1 ].seq.values
Y_first         = df_train[['display_id', 'seq']].drop_duplicates(subset='display_id', keep='first').seq.values
Y_ranks         = Y_seq - Y_first

# At this point, some simplification of the MAP function given what we know about this competition gives us this quick calc

score           = np.mean( 1.0 / (1.0 + Y_ranks) )

print("MAP: %.12f" % score)
