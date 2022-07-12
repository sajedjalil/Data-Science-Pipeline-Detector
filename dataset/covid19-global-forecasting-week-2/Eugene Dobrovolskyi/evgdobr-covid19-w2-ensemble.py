from datetime import datetime
from warnings import filterwarnings

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import ensemble
from xgboost import XGBRegressor

from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

filterwarnings('ignore')


def year_day_number(date):
    adate = datetime.strptime(date, "%Y-%m-%d")
    return adate.timetuple().tm_yday


train = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-2/test.csv")
submission_path = r"/kaggle/working/submission.csv"


train['Province_State'].fillna("Unknown", inplace=True)
test['Province_State'].fillna("Unknown", inplace=True)
train['Date'] = train['Date'].apply(year_day_number)
test['Date'] = test['Date'].apply(year_day_number)

df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in train.Country_Region.unique():
    for state in train.loc[train.Country_Region == country, :].Province_State.unique():
        print("Country {} - {}".format(country, state))
        x = train.loc[(train.Country_Region == country) & (train.Province_State == state),
                      ['Date', 'ConfirmedCases', 'Fatalities']]

        confirmed_cases = x.ConfirmedCases
        fatalities = x.Fatalities

        x = x.loc[:, ['Date']]
        x = np.log(x)

        submission = test.loc[(test.Country_Region == country) & (test.Province_State == state), ['Date', 'ForecastId']]
        submission_id = submission.loc[:, 'ForecastId']
        submission = submission.loc[:, ['Date']]
        submission = np.log(submission)

        gradient_boosting = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42,
                                                      loss='ls')
        xgb_regressor = XGBRegressor(n_estimators=100)
        ridgle = RidgeCV()
        linear = LinearSVR(random_state=42)
        random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

        estimators = [  
            ('gbr', gradient_boosting),
            ('xgb', xgb_regressor) #,
            # ('lr', ridgle),
            # ('svr', linear)
        ]

        confirmed_cases_predictor = StackingRegressor(estimators=estimators, final_estimator=random_forest)
        fatalities_predictor = StackingRegressor(estimators=estimators, final_estimator=random_forest)

        confirmed_cases_predictor.fit(x, confirmed_cases)
        fatalities_predictor.fit(x, fatalities)

        confirmed_cases_pred = confirmed_cases_predictor.predict(submission)
        fatalities_pred = fatalities_predictor.predict(submission)

        df = {'ForecastId': submission_id, 'ConfirmedCases': confirmed_cases_pred, 'Fatalities': fatalities_pred}
        df = pd.DataFrame(df)
        df_out = pd.concat([df_out, df], axis=0)

df_out.ForecastId = df_out.ForecastId.astype('int')
df_out.to_csv(submission_path, index=False)
print("Done. Saved to '{}'".format(submission_path))
