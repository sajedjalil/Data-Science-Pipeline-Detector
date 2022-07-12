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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

exclude_zero_values = True
days_for_shifting = 7  # be careful: do not change this parameter because column names depends on this number

train = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-4/train.csv")
test = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-4/test.csv")
submission_path = r"/kaggle/working/submission.csv"



# some helpful functions

# convert date to year day number
def year_day_number(date):
    adate = datetime.strptime(date, "%Y-%m-%d")
    return adate.timetuple().tm_yday / 10


def my_print(print_data):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(print_data)


# augment some data for prediction
def extend_train_data_for_country(country_group):
    country_group = country_group.reset_index().copy()

    first_case_date = pd.to_datetime(country_group[country_group.ConfirmedCases > 0].Date.min())
    first_fatality_date = pd.to_datetime(country_group[country_group.Fatalities > 0].Date.min())

    country_group['Week'] = pd.to_datetime(country_group.Date).apply(lambda d: int(d.strftime('%W'))).copy()
    country_group['Week_day'] = pd.to_datetime(country_group.Date).apply(lambda d: int(d.strftime('%u'))).copy()

    first_week = country_group[country_group.ConfirmedCases > 0].Week.min()
    running_week = country_group.groupby('Province_State').Week.apply(lambda w: w - first_week).values
    country_group['Running_week'] = np.where(running_week < 0, 0, running_week)

    Days_from_1_case = (pd.to_datetime(country_group.Date) - first_case_date).dt.days.fillna(0)
    country_group['Days_from_1_case'] = np.where(Days_from_1_case < 0, 0, Days_from_1_case)

    Days_from_1_fatality = (pd.to_datetime(country_group.Date) - first_fatality_date).dt.days.fillna(0)
    country_group['Days_from_1_fatality'] = np.where(Days_from_1_fatality < 0, 0, Days_from_1_fatality)

    country_group.drop(columns=['index', 'Week'], inplace=True)

    return country_group


# make shifts
def shifts(data_for_shifting):
    for column in ['ConfirmedCases', 'Fatalities']:
        for days in range(1, days_for_shifting + 1):
            new_column_name = column + "_shift_" + str(days)
            data_for_shifting[new_column_name] = data_for_shifting[column].shift(periods=days, fill_value=0)
    return data_for_shifting


# from exp to linear
train["ConfirmedCases"] = np.log1p(train["ConfirmedCases"])
train["Fatalities"] = np.log1p(train["Fatalities"])

#  extend test data to make it compatible with extend_train_data_for_country
test["ConfirmedCases"] = 0
test["Fatalities"] = 0
test["index"] = 0

# fill NA
test['Province_State'].fillna("Unknown", inplace=True)
train['Province_State'].fillna("Unknown", inplace=True)

# add shifts to use historical data for forecasting and augment some data
train = shifts(train)
train = pd.concat([extend_train_data_for_country(data)
                   for group, data in train.groupby(['Country_Region', 'Province_State'])
                   ])

# add shifts to use historical data for forecasting and augment some data
test = shifts(test)  # just to create columns, shifting will be filled later with forecasting
test = pd.concat([extend_train_data_for_country(data)
                  for group, data in test.groupby(['Country_Region', 'Province_State'])
                  ])

# convert dates to year_day_number
test['Date'] = test['Date'].apply(year_day_number)
train['Date'] = train['Date'].apply(year_day_number)

#  model training and forecasting
df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in train.Country_Region.unique():
    for state in train.loc[train.Country_Region == country, :].Province_State.unique():
        print("Country {} - {}".format(country, state))

        if exclude_zero_values:
            x = train.loc[(train.Country_Region == country) &
                          (train.Province_State == state) &
                          (train.ConfirmedCases > 0), :]
        else:
            x = train.loc[(train.Country_Region == country) & (train.Province_State == state), :]

        # prepare train data
        confirmed_cases = x.ConfirmedCases
        fatalities_cases = x.Fatalities
        x.drop(columns=['ConfirmedCases', 'Fatalities', 'Id', 'Province_State', 'Country_Region'], inplace=True)
        x.drop(columns=['Week_day', 'Running_week', "Days_from_1_case", "Days_from_1_fatality"], inplace=True)

        # prepare input forecast data
        submission = test.loc[(test.Country_Region == country) & (test.Province_State == state), :]
        submission_id = submission.loc[:, 'ForecastId']
        columns_to_drop = ['ForecastId', 'Province_State', 'Country_Region', 'level_0', 'ConfirmedCases', 'Fatalities']
        submission.drop(columns=columns_to_drop, inplace=True)

        confirmed_cases_predictor = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=15,
                                                              random_state=0,
                                                              loss='huber').fit(x, confirmed_cases)
        fatalities_predictor = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=15,
                                                         random_state=0,
                                                         loss='huber').fit(x, fatalities_cases)

        # # create stacking model
        # gradient_boosting = GradientBoostingRegressor(random_state=0)
        # xgb_regressor = XGBRegressor(n_estimators=100)
        # ridgle = RidgeCV()
        # linear = LinearSVR(random_state=42)
        # random_forest = RandomForestRegressor(n_estimators=100000, random_state=42)

        # params = {'gbr__n_estimators': [10, 100, 1000, 10000],
        #           'gbr__learning_rate': [0.001, 0.01, 0.1, 1, 3],
        #           'gbr__max_depth': [10, 15, 30]}
        #
        # params = {'gbr__n_estimators': [10, 100, 1000, 10000],
        #           'gbr__learning_rate': [0.001, 0.01, 0.1, 1, 3],
        #           'gbr__max_depth': [10, 15, 30]}

        # params = {'gbr__n_estimators': [1000, 3000, 5000],
        #           'gbr__learning_rate': [0.1, 0.01],
        #           'gbr__max_depth': [10, 15, 30],
        #           'gbr__loss': ['huber']}
        #
        # estimators = [('gbr', gradient_boosting)  # ,
        #               # ('xgb', xgb_regressor)  # ,
        #               # ('lr', ridgle),
        #               # ('svr', linear)
        #               ]


        # params = {'xgb__n_estimators': [1000],
        #           'xgb__learning_rate': [0.1]}
        #
        # estimators = [#('gbr', gradient_boosting)  # ,
        #     ('xgb', xgb_regressor)  # ,
        #     # ('lr', ridgle),
        #     # ('svr', linear)
        # ]


        # params = {}
        # estimators = [('svr', linear)]

        # confirmed_cases_predictor = StackingRegressor(estimators=estimators, final_estimator=random_forest)
        # fatalities_predictor = StackingRegressor(estimators=estimators, final_estimator=random_forest)
        # confirmed_cases_predictor = StackingRegressor(estimators=estimators)
        # fatalities_predictor = StackingRegressor(estimators=estimators)

        # # train model confirmed_cases_predictor
        # confirmed_cases_predictor_grid = GridSearchCV(estimator=confirmed_cases_predictor,
        #                                               param_grid=params,
        #                                               cv=5,
        #                                               refit=True,
        #                                               n_jobs=-1,
        #                                               verbose=0)
        # confirmed_cases_predictor_grid.fit(x, confirmed_cases)
        # confirmed_cases_predictor = confirmed_cases_predictor_grid.best_estimator_
        # print (confirmed_cases_predictor_grid.best_params_)
        #
        # # train model fatalities_predictor
        # fatalities_predictor_grid = GridSearchCV(estimator=fatalities_predictor,
        #                                          param_grid=params,
        #                                          cv=5,
        #                                          refit=True,
        #                                          n_jobs=-1,
        #                                          verbose=0)
        #
        # x = x.reindex(sorted(x.columns), axis=1)
        # fatalities_predictor_grid.fit(x, fatalities_cases)
        # fatalities_predictor = fatalities_predictor_grid.best_estimator_
        # print (fatalities_predictor_grid.best_params_)

        last_line = x.iloc[-1]

        # fill known shifts
        for day in range(2, days_for_shifting + 1):
            submission.ix[0, "ConfirmedCases_shift_" + str(day)] = last_line.loc["ConfirmedCases_shift_" + str(day - 1)]
            submission.ix[0, "Fatalities_shift_" + str(day)] = last_line.loc["Fatalities_shift_" + str(day - 1)]

        submission.ix[0, "ConfirmedCases_shift_" + str(1)] = confirmed_cases.iloc[-1]
        submission.ix[0, "Fatalities_shift_" + str(1)] = fatalities_cases.iloc[-1]

        # submission.ix[0, "Running_week"] = last_line.loc["Running_week"]
        # submission.ix[0, "Week_day"] = last_line.loc["Week_day"] + 1
        # submission.ix[0, "Days_from_1_case"] = last_line.loc["Days_from_1_case"] + 1
        # submission.ix[0, "Days_from_1_fatality"] = last_line.loc["Days_from_1_fatality"] + 1
        #
        # # check that we jump to the next week
        # if submission.ix[0, "Week_day"] == 8:
        #     submission.ix[0, "Week_day"] = 1
        #     submission.ix[0, "Running_week"] += 1

        #
        #
        # Let's do prediction
        #
        #
        # prepare output for country
        df_out_country = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

        submission.drop(columns=['Week_day', 'Running_week', "Days_from_1_case", "Days_from_1_fatality"], inplace=True)
        submission = submission.reindex(sorted(submission.columns), axis=1)
        for row_id in range(submission.shape[0]):  # we will do prediction for each row separately

            submission_for_prediction = np.array(submission.ix[row_id, :]).reshape(1, -1)


            confirmed_cases_prediction = confirmed_cases_predictor.predict(submission_for_prediction)
            fatalities_prediction = fatalities_predictor.predict(submission_for_prediction)

            # print("row {}".format(row_id))
            # print(submission_for_prediction)
            # print (confirmed_cases_prediction, fatalities_prediction)

            # fill next row of submission dataframe with obtained predicted data
            if row_id != submission.shape[0] - 1:
                # copy to the next row
                for j in range(2, days_for_shifting + 1):
                    submission.ix[row_id + 1, "ConfirmedCases_shift_" + str(j)] = \
                        submission.ix[row_id, "ConfirmedCases_shift_" + str(j - 1)]
                    submission.ix[row_id + 1, "Fatalities_shift_" + str(j)] = \
                        submission.ix[row_id, "Fatalities_shift_" + str(j - 1)]

                submission.ix[row_id + 1, "ConfirmedCases_shift_" + str(1)] = confirmed_cases_prediction
                submission.ix[row_id + 1, "Fatalities_shift_" + str(1)] = fatalities_prediction

                # submission.ix[row_id + 1, "Running_week"] = submission.ix[row_id, "Running_week"]
                # submission.ix[row_id + 1, "Week_day"] = submission.ix[row_id, "Week_day"] + 1
                # submission.ix[row_id + 1, "Days_from_1_case"] = submission.ix[row_id, "Days_from_1_case"] + 1
                # submission.ix[row_id + 1, "Days_from_1_fatality"] = submission.ix[row_id, "Days_from_1_fatality"] + 1
                #
                # # check that we jump to the next week
                # if submission.ix[row_id + 1, "Week_day"] == 8:
                #     submission.ix[row_id + 1, "Week_day"] = 1
                #     submission.ix[row_id + 1, "Running_week"] += 1

            df = pd.DataFrame({'ForecastId': submission_id[row_id],
                               'ConfirmedCases': np.exp(confirmed_cases_prediction) - 1,
                               'Fatalities': np.exp(fatalities_prediction) - 1})

            df_out_country = pd.concat([df_out_country, df], axis=0)

        df_out = pd.concat([df_out, df_out_country], axis=0)

df_out.ForecastId = df_out.ForecastId.astype('int')
df_out.to_csv(submission_path, index=False)
print("Done. Saved to '{}'".format(submission_path))
