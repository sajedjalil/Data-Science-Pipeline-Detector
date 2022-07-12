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
sub_1 = pd.read_csv('../input/ieee-simple-lgbm/submission.csv')

# Blend of two kernels with old features (0.9468)
sub_2 = pd.read_csv('../input/ieee-cv-options/submission.csv')

# Add new features lgbm with CV (0.9485)
sub_3 = pd.read_csv('../input/ieee-lgbm-with-groupkfold-cv/submission.csv')

# Add catboost (0.9407)
sub_4 = pd.read_csv('../input/ieee-catboost-baseline-with-groupkfold-cv/submission.csv')

# OLD (original kernel)
# This if just sum up all 
#sub_1['isFraud'] += sub_2['isFraud']
#sub_1['isFraud'] += sub_3['isFraud']
#sub_1['isFraud'] += sub_4['isFraud']

# Another approach. 
# Idea: Use weights based on proportionality with respect to each score

# Scores
score_1 = 0.9480
score_2 = 0.9468
score_3 = 0.9485
score_4 = 0.9407

# Sum
sum_scores = score_1 + score_2 + score_3 + score_4

# Weights with which we should sum all things up.
weight_1 = score_1/sum_scores
weight_2 = score_2/sum_scores
weight_3 = score_3/sum_scores
weight_4 = score_4/sum_scores


sub_1['isFraud'] = weight_1*sub_1['isFraud'] + weight_2*sub_2['isFraud'] + weight_3*sub_3['isFraud'] + weight_4*sub_4['isFraud']

# Attention: scores are similar => effect of approach is small, but positive (should be)!

sub_1.to_csv('submission.csv', index=False)

# CONCLUSION: 
# Effect is small because scores are almost identical...Sometimes will be no improvements. But take care!
# But if you deal with set of scores with 10-20% difference, the effect can be significant

# Good luck.