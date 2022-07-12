# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))
catboost_result=pd.read_csv('../input/catboostarter-ryan/submission.csv')
lgb_result=pd.read_csv('../input/good-fun-with-ligthgbm-meanenc-more-features/subm_lgbm_auc0.77965795.csv')

catboost_result.rename(columns={'TARGET':'cat_TARGET'},inplace=True)
lgb_result.rename(columns={'TARGET':'lgb_TARGET'},inplace=True)

# print(catboost_result)
# print(lgb_result)
sub=pd.merge(catboost_result,lgb_result,on='SK_ID_CURR')
sub['TARGET']=0*sub['cat_TARGET']+1*sub['lgb_TARGET']
sub=sub.drop(columns=['lgb_TARGET','cat_TARGET'])
sub.to_csv('subm_lgbm_auc.csv', index=False, float_format='%.8f')

# Any results you write to the current directory are saved as output.