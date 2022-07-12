# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
import matplotlib.dates as dates
from cycler import cycler
from scipy.ndimage.filters import gaussian_filter1d
from datetime import timedelta, date
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


##################################################################
# GLOBAL VARIABLES
##################################################################

sample_size = 3
# cluster 0: 22 yields 419k, 30 yileds 331k, 35 yields 290k, 40 yields 257k, 45 228k, 47 218k
train_history_days = 80
train_future_days = 34

# useful way to create an incrementing index df_forecast['forecast_sequence'] = np.arange(len(df_forecast))

##################################################################
# FILE OPERATIONS
##################################################################

# read in training data and test data
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', index_col=['Id'])
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv', index_col=['ForecastId'])
cluster_curves = pd.read_csv('/kaggle/input/jonty-covid-forecast/cluster_all.csv')
country_clusters = pd.read_csv(
    '/kaggle/input/jonty-covid-forecast/country_clusters_out_affinity.csv', index_col=['Geo'])
# clean
train["Province_State"].fillna("", inplace=True)
test["Province_State"].fillna("", inplace=True)
train["Geo"] = train['Country_Region'] + '_' + train['Province_State']
test["Geo"] = test['Country_Region'] + '_' + test['Province_State']
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
cluster_curves['Date'] = pd.to_datetime(cluster_curves['Date'])
test['ForecastId_'] = test.index

##################################################################
# DATE MANIPULATIONS
##################################################################

submission_date_min = test['Date'].min()
train_date_max = train['Date'].max()
future_date = train_date_max + timedelta(days=train_future_days)
history_date = train_date_max - timedelta(days=train_history_days)


# STEP 1
# CREATE SITE CLUSTER CONFIRMED RATIO AND FATALITY RATIO INTERFACE
# MOST RECENT 14 DAYS OF CLUSTER / 14 DAYS OF COUNTRY
# SAVE AS COUNTRY - CLUSTER - RATIO_CONFIRMED - RATIO_FATALITY

##################################################################
# GET RATIOS
##################################################################

# countries interface
countries_interface = pd.DataFrame(
    train.Geo.unique(), columns=['Geo']).reset_index()
countries_interface = pd.merge(countries_interface, country_clusters['cluster'], how='left', left_on=[
    'Geo'], right_on=['Geo'], left_index=True)

cluster_summary = cluster_curves[(
    cluster_curves["Date"] > history_date) & (cluster_curves["Date"] <= train_date_max)].reset_index()

cluster_summary_agg = cluster_summary.groupby(
    ['cluster']).sum().reset_index()

train_summary = train[
    train["Date"] > history_date].reset_index()

train_summary_agg = train_summary.groupby(
    ['Geo']).sum().reset_index()

countries_interface = pd.merge(countries_interface, cluster_summary_agg[['cluster', 'ConfirmedCases', 'Fatalities']], how='left', left_on=[
    'cluster'], right_on=['cluster'], left_index=True)

countries_interface = pd.merge(countries_interface, train_summary_agg[['Geo', 'ConfirmedCases', 'Fatalities']], how='left', left_on=[
    'Geo'], right_on=['Geo'], left_index=True)


# stick infintismal for zero
countries_interface[countries_interface["ConfirmedCases_y"] == 0].ConfirmedCases_y = 0.000000001
countries_interface[countries_interface["Fatalities_y"] == 0].Fatalities_y = 0.000000001
#print(countries_interface)

countries_interface['RATIO_CONFIRMED'] = countries_interface.ConfirmedCases_y / countries_interface.ConfirmedCases_x
countries_interface['RATIO_FATALITY'] = countries_interface.Fatalities_y / countries_interface.Fatalities_x
countries_interface = countries_interface[[
    'Geo', 'cluster', 'RATIO_CONFIRMED', 'RATIO_FATALITY']]
cluster_forecast = pd.merge(cluster_curves, countries_interface, on=['cluster'])

cluster_forecast = cluster_forecast[[
    'Geo', 'Date', 'cluster', 'ConfirmedCases',    'Fatalities', 'RATIO_CONFIRMED', 'RATIO_FATALITY']]
cluster_forecast['ConfirmedCases'] = cluster_forecast['ConfirmedCases'] * cluster_forecast['RATIO_CONFIRMED']
cluster_forecast['Fatalities'] = cluster_forecast['Fatalities'] * cluster_forecast['RATIO_FATALITY']
#print(cluster_forecast)


#cluster_forecast.to_csv('cluster_forecast.csv')

subsy = pd.merge(test, cluster_forecast, how = 'left', left_on = ['Date','Geo'], right_on = ['Date','Geo'])
#subsy = pd.merge(test, subsy, how = 'inner', left_on = ['Country_Region','Province_State','Date'], right_on = ['Country_Region','Province_State','Date'])


subsy = subsy[['ForecastId_','ConfirmedCases','Fatalities']]
subsy = subsy.rename(columns={'ForecastId_': 'ForecastId'})
new_row = {'ForecastId':13459, 'ConfirmedCases':100, 'Fatalities':12}
subsy = subsy.append(new_row, ignore_index=True)
subsy2 = pd.merge(submission, subsy, how = 'left', left_on = ['ForecastId'], right_on = ['ForecastId'])
subsy2 = subsy2[['ForecastId','ConfirmedCases_y','Fatalities_y']]
subsy2 = subsy2.rename(columns={'ConfirmedCases_y': 'ConfirmedCases'})
subsy2 = subsy2.rename(columns={'Fatalities_y': 'Fatalities'})
print(subsy2)
header = ['ForecastId','ConfirmedCases','Fatalities']
subsy2.to_csv('submission.csv', columns = header, index=False)
#cluster_forecast['ConfirmedCases'] = cluster_forecast['ConfirmedCases']
#cluster_forecast.to_csv('cluster_forecast_output.csv')
