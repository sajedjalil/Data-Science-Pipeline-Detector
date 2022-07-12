

# Original code is taken from this kernell: https://www.kaggle.com/meli19/ensemble-public-submissions
# I've just changed the weights a little bit.

# submission_1 https://www.kaggle.com/aharless/exclude-same-wk-res-from-nitin-s-surpriseme2-w-nn
# submission_2 https://www.kaggle.com/meli19/surprise-me-h2o-automl-version-ver5-lb-0-479
# submission_3 https://www.kaggle.com/nitinsurya/surprise-me-2-neural-networks-keras
# submission_4 https://www.kaggle.com/tejasrinivas/surprise-me-4-lb-0-479

# PLEASE think about the overfitting problem !!! take your own risk of using this kernel.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy.stats.mstats import gmean

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sub1 = pd.read_csv('../input/pubsubrrvf/submission_1.csv')
sub2 = pd.read_csv('../input/pubsubrrvf/submission_2.csv')
sub3 = pd.read_csv('../input/pubsubrrvf/submission_3.csv')
sub4 = pd.read_csv('../input/pubsubrrvf/submission_4.csv')


#concatenate target columns on the same Dataframe
preds = pd.concat([sub1['visitors'], sub2['visitors'], 
        sub3['visitors'], sub4['visitors']])
        
        
#Apply geometric mean 
preds = preds.groupby(level=0).apply(gmean)


# Create submission 
print(preds.head)
sub = pd.DataFrame()
sub['id'] = sub1['id']
sub['visitors'] = preds
	
sub.to_csv('sub_geometric.csv', index = False)
