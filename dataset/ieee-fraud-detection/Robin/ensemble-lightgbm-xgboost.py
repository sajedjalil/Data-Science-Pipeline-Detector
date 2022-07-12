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

#################################DataSet#################################
# https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm
# https://www.kaggle.com/kimchiwoong/simple-eda-ensemble-for-xgboost-and-lgbm

gbm_submission = pd.read_csv('/kaggle/input/feature-engineering-lightgbm/submission.csv')
print(gbm_submission.shape)

gbm_preds = gbm_submission['isFraud']

xgb_submission = pd.read_csv('/kaggle/input/simple-eda-ensemble-for-xgboost-and-lgbm/sample_submission_after_Feature_Engineering11.csv')
print(xgb_submission.shape)
xgb_preds = xgb_submission['isFraud']
xgb_preds.head()

 
final_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
final_submission['isFraud'] = gbm_preds * 0.8 + xgb_preds * 0.2
# final_submission['isFraud'] = (gbm_preds + xgb_preds)/2
final_submission.to_csv("submission.csv", index=False)