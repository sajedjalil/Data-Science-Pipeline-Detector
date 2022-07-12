# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

b1 = pd.read_csv('../input/submission/submission_3_seeds_lr002.csv').rename(columns={'TARGET':'dp1'})
b2 = pd.read_csv('../input/submission/submission_kernel02.csv').rename(columns={'TARGET':'dp2'})
b3 = pd.read_csv('../input/submission/tidy_xgb_0.78847.csv').rename(columns={'TARGET':'dp3'})
b1 = pd.merge(b1,b2,how='left', on='SK_ID_CURR')
b1 = pd.merge(b1,b3,how='left', on='SK_ID_CURR')

b1['TARGET'] = (b1['dp1'] * 0.1) + (b1['dp2'] * 0.6) + (b1['dp3'] * 0.3) 
b1[['SK_ID_CURR','TARGET']].to_csv('Submission_HomeCredit_Blend.csv', index=False)

