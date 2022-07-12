#forked from https://www.kaggle.com/muhakabartay/0-8518-what-proper-weights-give-ieee-int-blend



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# General imports
import pandas as pd
import os, sys, gc, warnings

warnings.filterwarnings('ignore')

## PLEASE UPVOTE THIS kernel AND ORIGINAL by Konstantin Yakovlev (https://www.kaggle.com/kyakovlev)
## original: https://www.kaggle.com/kyakovlev/ieee-internal-blend
## Also those kernels to which Konstantin's kernel is based on (see in original kernel). 

## SEE idea of these changes below

########################### DATA LOAD/MIX/EXPORT
#################################################################################

# Simple lgbm (0.9480)
sub_1 = pd.read_csv('../input/ieee-simple-lgbm/submission.csv',index_col=0)

# Blend of two kernels with old features (0.9468)
sub_2 = pd.read_csv('../input/ieee-cv-options/submission.csv', index_col=0)

# Add new features lgbm with CV (0.9485)
sub_3 = pd.read_csv('../input/ieee-lgbm-with-groupkfold-cv/submission.csv', index_col=0)

# Add catboost (0.9407)
sub_4 = pd.read_csv('../input/ieee-catboost-baseline-with-groupkfold-cv/submission.csv', index_col=0)

outs = [sub_1,sub_2,sub_3,sub_4]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "m" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
corr = concat_sub.iloc[:,1:].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
rank = np.tril(concat_sub.iloc[:,1:].corr().values,-1)
m = (rank>0).sum()
m_gmean, s = 0, 0

# OLD (original kernel)
# This if just sum up all 
#sub_1['isFraud'] += sub_2['isFraud']
#sub_1['isFraud'] += sub_3['isFraud']
#sub_1['isFraud'] += sub_4['isFraud']
for n in range(min(rank.shape[0],m)):
    mx = np.unravel_index(rank.argmin(), rank.shape)
    w = (m-n)/(m+n)
    print(w)
    m_gmean += w*(np.log(concat_sub.iloc[:,mx[0]+1])+np.log(concat_sub.iloc[:,mx[1]+1]))/2
    s += w
    rank[mx] = 1
m_gmean = np.exp(m_gmean/s)
# Another approach. 
# Idea: Use weights based on proportionality with respect to each score

# Scores
concat_sub['isFraud'] = m_gmean
concat_sub[['TransactionID','isFraud']].to_csv('stack_gmean.csv', 
                                        index=False, float_format='%.4g')

# Attention: scores are similar => effect of approach is small, but positive (should be)!



# CONCLUSION: 
# Effect is small because scores are almost identical...Sometimes will be no improvements. But take care!
# But if you deal with set of scores with 10-20% difference, the effect can be significant

# Good luck.