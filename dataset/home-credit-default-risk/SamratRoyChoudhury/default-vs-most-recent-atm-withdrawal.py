# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
# The following code is used to see the probability of default against the percentage of most recent atm withdrawl of the total creditcard usage

ccdf = pd.read_csv("../input/credit_card_balance.csv")
atdf = pd.read_csv("../input/application_train.csv")



ccbdf = ccdf.sort_values('MONTHS_BALANCE', ascending=False).groupby(['SK_ID_CURR'], as_index=False).first()

ccbdf['CSH_WD_PRCNT']= ccbdf['AMT_DRAWINGS_ATM_CURRENT'] / (ccbdf['AMT_DRAWINGS_CURRENT'] + ccbdf['AMT_DRAWINGS_OTHER_CURRENT'] + ccbdf['AMT_DRAWINGS_POS_CURRENT'])

sgdf = atdf[['SK_ID_CURR', 'TARGET']].merge(ccbdf[['SK_ID_CURR','CSH_WD_PRCNT']], on = 'SK_ID_CURR')

sgdf['CSH_WD_PRCNT'] = sgdf['CSH_WD_PRCNT'] * 10
sgdf = sgdf.dropna()

sns.lmplot(y="CSH_WD_PRCNT", x="TARGET", data=sgdf, fit_reg=False)
#print(sgdf)