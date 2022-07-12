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

#################################################################################
##                            PLEASE UPVOTE THIS kernel                        ##
#################################################################################

########################### DATA LOAD/MIX/EXPORT
#################################################################################



# Check you data output from different models and submission
# Note: This is example, no private data included (and only few samples)
# As example I put output from this kernel repeated 5 times, so that submission_N_SCORE.csv
# https://www.kaggle.com/roydatascience/light-gbm-with-complete-eda
# Please upvote it.
# Thus, put your output submissions' data or from others (in .csv), as more as you wish as soon as it gives better score. 
#!ls ../input/modelsdata
#!ls ../input/
#!ls ../input/modelsdata -1 | wc -l

suball = []

import glob
print(glob.glob("../input/yourallmodelsdata/*.csv"))

ll = len(glob.glob("../input/yourallmodelsdata/*.csv"))
print(ll)

for ii in range(0,ll):
    print(ii)
    #print(glob.glob("../input/modelsdata/*.csv")[0])
    suball.append(glob.glob("../input/yourallmodelsdata/*.csv")[ii])
    
print(suball)

tot = 0
# 40:46 -> we take submission value from SCORE in name of the file (you name it by yourself of course)
# Thus, check this depending of which name you give and which path is used
for i in range(0,ll):
    #print(suball[i][40:46])
    tot += float(suball[i][40:46])
        
print("SUM: ",tot) # sum all, later to divide on it and get weights for each submission value


df_suball = []
for i in range(0,ll):
    df_suball.append(0)
for i in range(0,ll):
    df_suball[i] = pd.read_csv(glob.glob("../input/yourallmodelsdata/*.csv")[i])
    
print(df_suball[0])

## Weights with respect to submission score
for i in range(1,ll):
    print(suball[i][40:46]) # for check 0.XXXX value should be
    df_suball[0]['isFraud'] += (float(suball[i][40:46])/tot)*df_suball[i]['isFraud']
    
## add one other nice kernel output
#!ls ../input/
df_subA = pd.read_csv("../input/gmean-of-low-correlation-lb-0-952x/stack_gmean.csv")
# go
df_suball[0]['isFraud'] += df_subA['isFraud']

df_suball[0].to_csv('submissionROW.csv', index=False)