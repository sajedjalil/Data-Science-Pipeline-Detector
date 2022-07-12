import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv("../input/train_V2.csv")

baseline_factors = [factor for factor in list(train_data) if factor not in ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']]
baseline_model = RandomForestRegressor(n_estimators = 1000)
baseline_model.fit(train_data.iloc[1:10000][baseline_factors], train_data.iloc[1:10000]['winPlacePerc'])

test_data = pd.read_csv("../input/test_V2.csv")
test_data['winPlacePerc'] = baseline_model.predict(test_data[baseline_factors])
baseline_submission = test_data[['Id', 'winPlacePerc']]
baseline_submission.to_csv('baseline_submission.csv', index=False)