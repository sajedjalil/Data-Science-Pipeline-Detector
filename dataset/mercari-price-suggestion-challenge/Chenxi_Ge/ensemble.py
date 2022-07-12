import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# my first kernel, feels good.

# Ridge LBGM:
#    https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44823
# Average of lightgbm +ridge 0.45704LB :
#    https://www.kaggle.com/yliu9999/average-of-lightgbm-ridge-0-45704lb



f1 = pd.read_csv('../input/mercariuserresult/submission_lgbm_ridge_5.csv')
f2 = pd.read_csv('../input/mercariuserresult/submission.csv')

f3 = f1.copy()
f3.price = f1.price * 0.7 + f2.price * 0.3

f3.to_csv('my_submit.csv', header=True, index=False)