import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TEST_DATA_PATH = '../input/test_V2.csv'
TRAIN_DATA_PATH = "../input/train_V2.csv"


def change_columns_to_num(input_data, columns_from, columns_new):
    for column in columns_new:
        input_data[column] = np.where(input_data[columns_from] == column, 1, 0)
    input_data = input_data.drop(columns_from, axis=1)
    return input_data


data = pd.read_csv(TRAIN_DATA_PATH)
data = data.dropna(axis=0)

y = data.winPlacePerc
X = data.copy()
X = change_columns_to_num(X, "matchType", ["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"])
X = X.drop(["Id", "groupId", "matchId", "winPlacePerc"], axis=1)

final_model = RandomForestRegressor(n_jobs=-1)
final_model.fit(X, y)

test_data = pd.read_csv(TEST_DATA_PATH)
test_data_Id = test_data.Id

test_X = change_columns_to_num(test_data, "matchType", ["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"])
test_X = test_X.drop(["Id", "groupId", "matchId"], axis=1)

test_predict_result = final_model.predict(test_X)

output = pd.DataFrame({'Id': test_data_Id, 'winPlacePerc': test_predict_result})
output.to_csv('submission.csv', index=False)
