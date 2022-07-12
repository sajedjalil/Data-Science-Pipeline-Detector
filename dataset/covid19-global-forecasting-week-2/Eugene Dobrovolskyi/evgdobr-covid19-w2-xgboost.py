from datetime import datetime
from warnings import filterwarnings

import numpy as np
import pandas as pd
from sklearn import linear_model
from xgboost import XGBRegressor

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

        confirmed_cases_pred1 = XGBRegressor(n_estimators=1000).fit(x, confirmed_cases).predict(submission)
        fatalities_pred1 = XGBRegressor(n_estimators=1000).fit(x, fatalities).predict(submission)

        confirmed_cases_predictor = linear_model.Ridge().fit(x, confirmed_cases)
        confirmed_cases_pred2 = confirmed_cases_predictor.predict(submission)

        fatalities_predictor = linear_model.Ridge().fit(x, fatalities)
        fatalities_pred2 = fatalities_predictor.predict(submission)

        confirmed_cases_pred = np.mean([confirmed_cases_pred1, confirmed_cases_pred2])
        fatalities_pred = np.mean([fatalities_pred1, fatalities_pred2])

        df = {'ForecastId': submission_id, 'ConfirmedCases': confirmed_cases_pred, 'Fatalities': fatalities_pred}
        df = pd.DataFrame(df)
        df_out = pd.concat([df_out, df], axis=0)

df_out.ForecastId = df_out.ForecastId.astype('int')
df_out.to_csv(submission_path, index=False)
print("Done. Saved to '{}'".format(submission_path))
