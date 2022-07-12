from datetime import datetime
from warnings import filterwarnings

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

filterwarnings('ignore')

exclude_zero_values = False


def year_day_number(date):
    adate = datetime.strptime(date, "%Y-%m-%d")
    return adate.timetuple().tm_yday


train = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-3/train.csv")
test = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-3/test.csv")
submission_path = r"/kaggle/working/submission.csv"


train['Province_State'].fillna("Unknown", inplace=True)
test['Province_State'].fillna("Unknown", inplace=True)
train['Date'] = train['Date'].apply(year_day_number)
test['Date'] = test['Date'].apply(year_day_number)

df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in train.Country_Region.unique():
    for state in train.loc[train.Country_Region == country, :].Province_State.unique():
        print("Country {} - {}".format(country, state))

        if exclude_zero_values:
            x = train.loc[(train.Country_Region == country) &
                          (train.Province_State == state) &
                          (train.ConfirmedCases > 0),  # don't take into account zero values
                          ['Date', 'ConfirmedCases', 'Fatalities']]
        else:
            x = train.loc[(train.Country_Region == country) & (train.Province_State == state),
                          ['Date', 'ConfirmedCases', 'Fatalities']]

        confirmed_cases = np.log1p(x.ConfirmedCases)
        fatalities_cases = np.log1p(x.Fatalities)

        x = x.loc[:, ['Date']]
        # x = np.log(x)
        # x = np.exp(x)


        submission = test.loc[(test.Country_Region == country) & (test.Province_State == state), ['Date', 'ForecastId']]
        submission_id = submission.loc[:, 'ForecastId']
        submission = submission.loc[:, ['Date']]
        # submission = np.log(submission)
        # submission = np.exp(submission)

        gradient_boosting = GradientBoostingRegressor(random_state=42, loss='ls')
        xgb_regressor = XGBRegressor(n_estimators=1000)
        ridgle = RidgeCV()
        linear = LinearSVR(random_state=42)
        random_forest = RandomForestRegressor(n_estimators=10, random_state=42)

        estimators = [('gbr', gradient_boosting),
                      # ('xgb', xgb_regressor)  # ,
                      # ('lr', ridgle),
                      # ('svr', linear)
                      ]

        confirmed_cases_predictor = StackingRegressor(estimators=estimators, final_estimator=random_forest)
        fatalities_predictor = StackingRegressor(estimators=estimators, final_estimator=random_forest)

        params = {'gbr__n_estimators': [10, 100, 1000, 10000],
                  'gbr__learning_rate': [0.001, 0.01, 0.1, 1, 3],
                  'gbr__max_depth': [1, 3]}

        #################
        confirmed_cases_predictor_grid = GridSearchCV(estimator=confirmed_cases_predictor,
                                                      param_grid=params,
                                                      cv=5,
                                                      refit=True,
                                                      n_jobs=-1,
                                                      verbose=0)

        confirmed_cases_predictor_grid.fit(x, confirmed_cases)
        confirmed_cases_predictor = confirmed_cases_predictor_grid.best_estimator_
        print (confirmed_cases_predictor_grid.best_params_)

        #################


        fatalities_predictor_grid = GridSearchCV(estimator=fatalities_predictor,
                                                 param_grid=params,
                                                 cv=5,
                                                 refit=True,
                                                 n_jobs=-1,
                                                 verbose=0)

        fatalities_predictor_grid.fit(x, fatalities_cases)
        fatalities_predictor = fatalities_predictor_grid.best_estimator_
        print (fatalities_predictor_grid.best_params_)

        #################

        confirmed_cases_prediction = np.exp(confirmed_cases_predictor.predict(submission))
        fatalities_prediction = np.exp(fatalities_predictor.predict(submission))

        df = {'ForecastId': submission_id, 'ConfirmedCases': confirmed_cases_prediction, 'Fatalities': fatalities_prediction}
        df = pd.DataFrame(df)
        df_out = pd.concat([df_out, df], axis=0)

df_out.ForecastId = df_out.ForecastId.astype('int')
df_out.to_csv(submission_path, index=False)
print("Done. Saved to '{}'".format(submission_path))
