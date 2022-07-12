import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from sklearn import linear_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

warnings.filterwarnings("ignore")

train_data_path = r"data\train.csv"
test_data_path = r"data\test.csv"
submission_data_path = r"data\submission.csv"

train_data_path = r"data\train_orig.csv"
test_data_path = r"data\test_orig.csv"
submission_data_path = r"data\submission.csv"

train_data_path = r"/kaggle/input/covid19-global-forecasting-week-2/train.csv"
test_data_path = r"/kaggle/input/covid19-global-forecasting-week-2/test.csv"
submission_data_path = r"/kaggle/working/submission.csv"


MODELS = dict()


def year_day_number(date):
    adate = datetime.strptime(date, "%Y-%m-%d")
    day_of_year = adate.timetuple().tm_yday
    return day_of_year


class NeuralNetwork:
    NUM_EPOCHS = 10000
    BATCH_SIZE = 64

    def __init__(self):
        self.nn_model = Sequential()

        self.nn_model.add(Dense(32, input_shape=(1,), name="fc1", activation="relu"))
        self.nn_model.add(Dense(168, name="fc2", activation="relu"))
        self.nn_model.add(Dense(8, name="fc3", activation="relu"))
        self.nn_model.add(Dense(1, name="fc4", activation="linear"))

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-07)

        self.nn_model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_error', 'accuracy'])

    def train(self, x, y):
        x = np.log(x)
        self.nn_model.fit(x, y, validation_data=(x, y),
                          batch_size=NeuralNetwork.BATCH_SIZE,
                          epochs=NeuralNetwork.NUM_EPOCHS,
                          verbose=2)

    def predict(self, x):
        x = np.log(x)
        return self.nn_model.predict(x)


class LocationPredictor:
    def __init__(self, data):
        data["id"] = [year_day_number(date) for date in data['Date']]  # add row id = year_day_number

        self.location = pd.unique(data["Location"])

        ## sklearn exponential regression
        self.confirmed_cases_predictor = linear_model.Ridge(alpha=1.0)
        self.confirmed_cases_predictor.fit(X=np.log(data[["id"]]), y=data["ConfirmedCases"])

        self.fatalities_predictor = linear_model.Ridge(alpha=1.0)
        self.fatalities_predictor.fit(X=np.log(data[["id"]]), y=data["Fatalities"])

        ##########################################
        ## xgboost exponential regression
        # params = {"objective": "reg:squarederror",
        #           "eval_metric": "rmse",
        #           "booster": "gblinear"}
        # 
        # df = pd.DataFrame({'x': np.log(data["id"]), 'y': data["ConfirmedCases"]})
        # train_xgb = xgb.DMatrix(df.drop('y', axis=1), df['y'])
        # self.confirmed_cases_predictor = xgb.train(dtrain=train_xgb, params=params)
        # 
        # df = pd.DataFrame({'x': np.log(data["id"]), 'y': data["Fatalities"]})
        # train_xgb = xgb.DMatrix(df.drop('y', axis=1), df['y'])
        # self.fatalities_predictor = xgb.train(dtrain=train_xgb, params=params)

        ##########################################

        ## NN model
        # self.confirmed_cases_predictor = NeuralNetwork()
        # self.confirmed_cases_predictor.train(x=data[["id"]], y=data["ConfirmedCases"])
        #
        # self.fatalities_predictor = NeuralNetwork()
        # self.fatalities_predictor.train(x=data[["id"]], y=data["Fatalities"])

    def predict(self, x):
        x = np.log(x)
        # confirmed_cases = self.confirmed_cases_predictor.predict(xgb.DMatrix(pd.DataFrame({'x': [x]})))
        # fatalities = self.fatalities_predictor.predict(xgb.DMatrix(pd.DataFrame({'x': [x]})))

        confirmed_cases = self.confirmed_cases_predictor.predict([[x]])
        fatalities = self.fatalities_predictor.predict([[x]])
        return confirmed_cases[0], fatalities[0]


def read_input_data():
    assert os.path.isfile(train_data_path), "{} doesn't exist".format(train_data_path)
    assert os.path.isfile(test_data_path), "{} doesn't exist".format(test_data_path)

    train_data = pd.read_csv(train_data_path)
    submission_data = pd.read_csv(test_data_path)

    # fill NAN data for specific columns
    for data in [train_data, submission_data]:
        for column in ["Country_Region", "Province_State"]:
            data[column] = data[column].fillna("Unknown")

    # create location column and convert data
    for data in [train_data, submission_data]:
        data["Location"] = data["Country_Region"] + "_" + data["Province_State"]
        data["Data"] = pd.to_datetime(data["Date"])

    train = train_data[["Date", "Location", "ConfirmedCases", "Fatalities"]].copy()
    test = train_data[["Date", "Location", "ConfirmedCases", "Fatalities"]].copy()
    submission = submission_data[["Date", "Location"]].copy()

    return train, test, submission


def run_pipeline():
    train, test, submission = read_input_data()

    # create and train models
    print(">>> Model training ... ")

    for location in tqdm.tqdm(pd.unique(train["Location"])):
        data = train[train["Location"] == location]
        MODELS[location] = LocationPredictor(data)

    print(">>> All models are trained :)")

    # make prediction
    print(">>> Predicting ... ")

    output = pd.DataFrame()
    counter = 1
    for location in tqdm.tqdm(pd.unique(submission["Location"])):
        submission_by_location = submission.loc[submission["Location"] == location]
        model = MODELS[location]

        for date in submission_by_location["Date"]:
            confirmed_cases, fatalities = model.predict(year_day_number(date))
            output_tmp = pd.DataFrame(
                {"ForecastId": [counter], "ConfirmedCases": [confirmed_cases], "Fatalities": [fatalities]}
            )

            output = pd.concat([output, output_tmp])
            counter += 1

            # save submission
    output = output[["ForecastId", "ConfirmedCases", "Fatalities"]]
    output.to_csv(submission_data_path, index=None)

    print(">>> Saved to {}.".format(submission_data_path))


if __name__ == "__main__":
    run_pipeline()
