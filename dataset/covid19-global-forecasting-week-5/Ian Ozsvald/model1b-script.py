# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# KAGGLE

# This is a first submission to the Covid 19 Kaggle prediction competition
# It uses the dumbest model I could come up with in an evening and a bit
# Hopefully I got the submission format right and I can iterate next week :-)
# This code is bloody awful, sorry to anyone who reads it.

import time
import tqdm

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

import pandas as pd

def clean_up(df):
    df['Date'] = pd.to_datetime(df.Date, dayfirst=False)
    # make queries a bit faster and save some RAM
    df['Country_Region'] = df['Country_Region'].astype('category')
    df['Province_State'] = df['Province_State'].astype('category')
    df['County'] = df['County'].astype('category')
    df['Target'] = df['Target'].astype('category')
    return df


# kaggle
train_filename = "../input/covid19-global-forecasting-week-5/train.csv"
test_filename = "../input/covid19-global-forecasting-week-5/test.csv"
submission_filename = "../input/covid19-global-forecasting-week-5/submission.csv"

df_train = pd.read_csv(train_filename).fillna('-')
df_train = clean_up(df_train)
df_test = pd.read_csv(test_filename).fillna('-')
df_test = clean_up(df_test)
print(df_train.shape, df_test.shape)

# normally df_train has Train: 2020-01-23 00:00:00 2020-05-07 00:00:00
#df_train = df_train.query('Date > "2020-04-20"')

print("Train:", df_train.Date.min(), df_train.Date.max(), df_train.shape)
print("Test:", df_test.Date.min(), df_test.Date.max(), df_test.shape)

df_submission = pd.read_csv(submission_filename)

# KAGGLE
    
class Model1b():
    def fit(self, X, y):
        """Assumes index is Date"""
        # take recent days, take the mean
        #self.most_recent_value = float(y[-5:].median())
        self.most_recent_value = float(y[-5:].mean())
            
    def predict(self, X):
        y = np.repeat(self.most_recent_value, X.shape[0])
        return y
    
est1b = Model1b()
cols = ['Weight', 'Population', 'Date']

est = est1b

# KAGGLE
unique_county_province_country_df = df_train[['County', 'Province_State', 'Country_Region']].drop_duplicates()
print(unique_county_province_country_df.shape)
unique_county_province_country_df.head()

# KAGGLE 

df_test['TargetValue'] = 0.0
t1 = time.time()
# note categories seem to give 25% speedup

# iterate over all of the query sets
# take recent timeseries, build silly model, make prediction from it
# write prediction into df_test
DEBUG = False
nbr_combinations = unique_county_province_country_df.shape[0]
print(f"Iterating over {nbr_combinations} location combinations")
for location_row in tqdm.tqdm(list(range(nbr_combinations))):
    #print(location_row, unique_county_province_country_df.iloc[location_row])
    #if location_row == 50:
    #    break
    row = unique_county_province_country_df.iloc[location_row]
    county = row['County']
    province = row['Province_State']
    country = row['Country_Region']
    for target_type in ['Fatalities', 'ConfirmedCases']:
        df_subset_train = df_train.query("County==@county and Province_State==@province and Country_Region==@country and Target==@target_type")
        df_subset_test = df_test.query("County==@county and Province_State==@province and Country_Region==@country and Target==@target_type")
        est.fit(df_subset_train[cols], df_subset_train['TargetValue'])
        y_pred = est.predict(df_subset_test[cols])

        predictions = pd.Series(y_pred, index=df_subset_test.index)
        df_test.loc[df_subset_test.index, 'TargetValue'] = predictions
        if location_row < 10 and DEBUG:
            print(target_type, county, province, country, predictions.iloc[-1])
    
t2 = time.time()
print(f"Took {t2-t1:0.2f} seconds")

# KAGGLE
df_005 = pd.DataFrame({'ForecastId_Quantile': df_test.ForecastId.apply(lambda n: f"{n}_0.05").values, 'TargetValue': df_test.TargetValue * 0.25})
df_050 = pd.DataFrame({'ForecastId_Quantile': df_test.ForecastId.apply(lambda n: f"{n}_0.5").values, 'TargetValue': df_test.TargetValue})
df_095 = pd.DataFrame({'ForecastId_Quantile': df_test.ForecastId.apply(lambda n: f"{n}_0.95").values, 'TargetValue': df_test.TargetValue * 1.75})
df_submission_new = pd.concat((df_005, df_050, df_095))
# note mergesort is only stable sort option
df_submission_new = df_submission_new.sort_index(kind="mergesort")
df_submission_new = df_submission_new.reset_index(drop=True)

df_submission_new.to_csv("submission.csv", index=False)

df_submission_new.head()