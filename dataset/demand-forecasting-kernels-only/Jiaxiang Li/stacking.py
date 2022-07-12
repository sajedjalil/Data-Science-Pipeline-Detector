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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

# Input

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
sample_submission  = pd.read_csv("../input/sample_submission.csv")
from sklearn.model_selection import train_test_split
validation_train, validation_test = train_test_split(train, test_size=0.3, random_state=123)

# Start stacking
# Split train data into two parts
part_1, part_2 = train_test_split(validation_train, test_size=0.5, random_state=123)
features = ['store', 'item']

# Train a Gradient Boosting model on Part 1
# gb = GradientBoostingRegressor().fit(part_1[features], part_1.sales)
gb = XGBRegressor().fit(part_1[features], part_1.sales)

# Train a Random Forest model on Part 1
rf = RandomForestRegressor().fit(part_1[features], part_1.sales)

# Make predictions on the Part 2 data
part_2['gb_pred'] = gb.predict(part_2[features])
part_2['rf_pred'] = rf.predict(part_2[features])

# Make predictions on the test data
test['gb_pred'] = gb.predict(test[features])
test['rf_pred'] = rf.predict(test[features])

from sklearn.linear_model import LinearRegression

# Create linear regression model without the intercept
lr = LinearRegression(fit_intercept=False)

# Train 2nd level model in the part_2 data
lr.fit(part_2[['gb_pred', 'rf_pred']], part_2.sales)

# Assign naive prediction to all the holdout observations
validation_test['gb_pred'] = gb.predict(validation_test[features])
validation_test['rf_pred'] = rf.predict(validation_test[features])
validation_test['pred'] = lr.predict(validation_test[['gb_pred', 'rf_pred']])

# Measure the local RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(validation_test['sales'], validation_test['pred']))
print('Validation RMSE for Stacking model: {:.3f}'.format(rmse))

# Make stacking predictions on the test data
test['sales'] = lr.predict(test[['gb_pred', 'rf_pred']])

# Look at the model coefficients
print(lr.coef_)
# If you see some coefficient very small, the stacking model is inefficient. 

print(test['sales'].head())
test[['id','sales']].to_csv("stacking_v1.0.0.csv", index=False)