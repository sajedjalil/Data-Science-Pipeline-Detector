#%% imports
import os
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error, precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import ShuffleSplit, cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

# constants
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
n_jobs = -1
colors = {
  'very_light_gray': '#ececec',
  'light_gray': '#b6b6b6',
  'medium_gray': '#929292',
  'very_dark_gray': '#414141',
  'orange': '#ff6f00',
  'light_orange': '#ffc090',
  'light_blue': '#79c3ff',
  'light_purple': '#d88aff',
  'light_green': '#b4ec70',
  'light_yellow': '#fff27e',
  'light_red': '#ff7482',
  'light_cyan': '#84ffff'
}

#%% load data
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv', parse_dates=['Date'])

print('Original training data')
# display(train)

start_date, cutoff_date, max_date = train['Date'].to_numpy()[0], np.datetime64('2020-03-24'), test['Date'].to_numpy()[-1]
data = train[train['Date'] <= cutoff_date].copy()
validation_data = train[train['Date'] >= cutoff_date].copy()
validation_data['ConfirmedCasesActual'] = validation_data['ConfirmedCases']
validation_data['ConfirmedCases'] = np.NaN
validation_data['FatalitiesActual'] = validation_data['Fatalities']
validation_data['Fatalities'] = np.NaN
dates, validation_dates, test_dates = data['Date'].unique(), validation_data['Date'].unique(), test['Date'].unique()

data['ForecastId'] = -1
validation_data['ForecastId'] = -1
test_data = test.copy()
test_data['ConfirmedCases'] = np.NaN

#%% create unique area key
def unique_area_key (row):
  if pd.isnull(row['Province/State']): return row['Country/Region']
  return f'{row["Country/Region"]} {row["Province/State"]}'
data['unique_area_key'] = data.apply(unique_area_key, axis=1)
validation_data['unique_area_key'] = validation_data.apply(unique_area_key, axis=1)
test_data['unique_area_key'] = test_data.apply(unique_area_key, axis=1)

#%% incorporate lag variables
def calculate_lag_columns(df, lag_list, column):
  df.sort_values(by=['unique_area_key', 'Date'], axis=0, inplace=True)
  for lag in lag_list:
    new_col_name = f'{column}_{str(lag)}'
    df[new_col_name] = np.NaN
    for key in df['unique_area_key'].unique():
      loc = df['unique_area_key'] == key
      df.loc[loc, new_col_name] = df.loc[loc, column].shift(lag, fill_value=0)
  return df

lag_list = [1, 2, 3, 4, 6, 8, 10, 12]

data = calculate_lag_columns(data, lag_list, 'ConfirmedCases')
data = calculate_lag_columns(data, lag_list, 'Fatalities')

#%% Incorporate world country data

# load world data
world_population = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

# select desired columns and rename some of them
world_population = world_population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]
world_population.columns = ['Country (or dependency)', 'population', 'density', 'land_area', 'med_age', 'urban_pop']

# Replace country name mismatches
world_population.loc[world_population['Country (or dependency)']=='Congo', 'Country (or dependency)'] = 'Republic of the Congo'
world_population.loc[world_population['Country (or dependency)']=='Czech Republic (Czechia)', 'Country (or dependency)'] = 'Czechia'
world_population.loc[world_population['Country (or dependency)']=="Côte d'Ivoire", 'Country (or dependency)'] = "Cote d'Ivoire"
world_population.loc[world_population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'
world_population.loc[world_population['Country (or dependency)']=='South Korea', 'Country (or dependency)'] = 'Korea, South'
world_population.loc[world_population['Country (or dependency)']=='Bahamas', 'Country (or dependency)'] = 'The Bahamas'
world_population.loc[world_population['Country (or dependency)']=='Taiwan', 'Country (or dependency)'] = 'Taiwan*'
gambia = world_population[world_population['Country (or dependency)']=='Gambia'].copy()
world_population.loc[world_population['Country (or dependency)']=='Gambia', 'Country (or dependency)'] = 'The Gambia'
world_population = world_population.append(gambia)
world_population.loc[world_population['Country (or dependency)']=='Gambia', 'Country (or dependency)'] = 'Gambia, The'

# Remove the % character from Urban Pop values
world_population['urban_pop'] = world_population['urban_pop'].str.rstrip('%')

# Transform Urban Pop and Med Age columns to floats
world_population.loc[world_population['urban_pop']=='N.A.', 'urban_pop'] = np.NaN
world_population['urban_pop'] = world_population['urban_pop'].astype(np.float)
world_population.loc[world_population['med_age']=='N.A.', 'med_age'] = np.NaN
world_population['med_age'] = world_population['med_age'].astype(np.float)

print('Cleaned country details dataset')
# display(world_population)

# join world data
data = data.merge(world_population, left_on='Country/Region', right_on='Country (or dependency)', how='left')
data = data.drop('Country (or dependency)', axis=1)
validation_data = validation_data.merge(world_population, left_on='Country/Region', right_on='Country (or dependency)', how='left')
validation_data = validation_data.drop('Country (or dependency)', axis=1)
test_data = test_data.merge(world_population, left_on='Country/Region', right_on='Country (or dependency)', how='left')
test_data = test_data.drop('Country (or dependency)', axis=1)

print('Joined data')
# display(data)

#%% Data pipeline
num_pipeline = Pipeline([
  ('median_imputer', SimpleImputer(strategy='median')),
  ('log_scaler', FunctionTransformer(np.log1p)),
  ('std_scaler', StandardScaler())
])
regular_num_pipeline = Pipeline([
  ('median_imputer', SimpleImputer(strategy='median')),
  ('std_scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
  ('categorize', OneHotEncoder(drop=None, sparse=False))
])
pipeline = ColumnTransformer([
  ('regular_num', regular_num_pipeline, ['Lat', 'Long', 'med_age']),
  ('numbers', num_pipeline, ['population', 'density', 'land_area', 'urban_pop']),
  ('lags', regular_num_pipeline, [f'ConfirmedCases_{lag}' for lag in lag_list]),
  ('categorical', categorical_pipeline, ['Country/Region', 'unique_area_key']),
], sparse_threshold=0, remainder='drop')
pipeline_fatality = ColumnTransformer([
  ('regular_num', regular_num_pipeline, ['Lat', 'Long', 'med_age']),
  ('numbers', num_pipeline, ['population', 'density', 'land_area', 'urban_pop']),
  ('lags', regular_num_pipeline, [f'Fatalities_{lag}' for lag in lag_list]),
  ('categorical', categorical_pipeline, ['Country/Region', 'unique_area_key']),
], sparse_threshold=0, remainder='drop')

#%% Subset data
def subset_criteria (df, country_region=[], province_state=[], invert=False):
  if not isinstance(country_region, list): country_region = [country_region]
  if not isinstance(province_state, list): province_state = [province_state]
  if pd.isnull(country_region).all():
    return np.full(len(df), True)
  if pd.isnull(province_state).all():
    return np.isin(df['Country/Region'], country_region, invert=invert)
  return (np.isin(df['Country/Region'], country_region, invert=invert)) & (np.isin(df['Province/State'], province_state, invert=invert))

country_region = None # ['China', 'Korea, South', 'Singapore'] # None # 'China' # None # 'Germany' #'US' # 'China'
province_state = None # 'Hong Kong' # None # 'Hubei' # None #'Washington'

subset_data = data[subset_criteria(data, country_region, province_state, invert=True)].copy().reset_index(drop=True)
subset_validation_data = validation_data[subset_criteria(validation_data, country_region, province_state, invert=True)].copy().reset_index(drop=True)
subset_test_data = test_data[subset_criteria(test_data, country_region, province_state, invert=True)].copy().reset_index(drop=True)

#%% Run model
X = pipeline.fit_transform(subset_data)
y = subset_data['ConfirmedCases']
print(f'Data prepared shape: {X.shape}, y shape: {y.shape}')

X_fatality = pipeline_fatality.fit_transform(subset_data)
y_fatality = subset_data['Fatalities']
print(f'Data prepared shape (fatalities): {X_fatality.shape}, y shape: {y_fatality.shape}')

#
# Hyperparameter search
#
# model_pipeline = Pipeline([
#   ('model', XGBRegressor(random_state=42, n_jobs=n_jobs))
# ])
# param_grid = [
#   {
#     'model__reg_alpha': [0.001, 0.005, 0.01],
#     'model__reg_lambda': [0.001, 0.005, 0.01],
#     'model__max_depth': [3, 4, 5],
#     'model__learning_rate': [0.05, 0.1, 0.2],
#     'model__n_estimators': [100, 200, 275, 300, 325, 350]
#   }
# ]
# grid_search = GridSearchCV(
#   estimator=model_pipeline,
#   param_grid=param_grid,
#   cv=5,
#   verbose=True,
#   n_jobs=n_jobs
# )
# grid_search.fit(X, y)
# model = grid_search.best_estimator_
# print('Best parameters:')
# print(grid_search.best_params_)

# Best model parameters
model = XGBRegressor(
  n_estimators=200,
  max_depth=4,
  learning_rate=0.1,
  random_state=42,
  reg_alpha=0.001,
  reg_lambda=0.1,
  n_jobs=n_jobs
)
print('Fitting ConfirmedCases model')
model.fit(X, y)
model_pred = np.round(model.predict(X))

model_fatality = XGBRegressor(
  n_estimators=200,
  max_depth=4,
  learning_rate=0.1,
  random_state=42,
  reg_alpha=0.001,
  reg_lambda=0.1,
  n_jobs=n_jobs
)
print('Fitting Fatalities model')
model_fatality.fit(X_fatality, y_fatality)
model_pred_fatality = np.round(model_fatality.predict(X_fatality))

#
# Other models that were tried
#
# model = LinearRegression(n_jobs=n_jobs)
# model = Ridge(random_state=42)
# model = SGDRegressor(random_state=42, penalty='l1')
# best linear regressor
# model = ElasticNet(random_state=42, l1_ratio=0.9, alpha=4.0)
# model = MLPRegressor(random_state=42, hidden_layer_sizes=(100,))
# model = DecisionTreeRegressor(random_state=42)

#%% march forward in time starting at the first date
pred_df = subset_data.copy()
y_pred = model_pred.copy()
y_pred_fatality = model_pred_fatality.copy()
for date in subset_test_data[subset_test_data['Date'] < cutoff_date]['Date'].unique():
  pred_df.loc[pred_df['Date'] == date, 'ForecastId'] = subset_test_data[subset_test_data['Date'] == date]['ForecastId'].to_numpy()
for date in subset_test_data[subset_test_data['Date'] >= cutoff_date]['Date'].unique():
  print(f'bootstrapping for date {date}')
  df = pred_df.append(subset_test_data[subset_test_data['Date'] == date].copy(), ignore_index=True)
  df.sort_values(by=['unique_area_key', 'Date'], axis=0, inplace=True)
  df = df[pred_df.columns]
  df = calculate_lag_columns(df, lag_list, 'ConfirmedCases')
  df = calculate_lag_columns(df, lag_list, 'Fatalities')
  y_pred = np.round(model.predict(pipeline.transform(df[df['Date'] == date])))
  df.loc[df['Date'] == date, 'ConfirmedCases'] = y_pred
  y_pred_fatality = np.round(model_fatality.predict(pipeline_fatality.transform(df[df['Date'] == date])))
  df.loc[df['Date'] == date, 'Fatalities'] = y_pred_fatality
  pred_df = df

#%% plot for
def plot_for (column, subset_data, model_pred, pred_df, subset_validation_data, country_region, province_state=None):
  p_data = subset_data.loc[subset_criteria(subset_data, country_region, province_state)]
  p_pred = pred_df.loc[subset_criteria(pred_df, country_region, province_state)]
  p_valid = subset_validation_data.loc[subset_criteria(subset_validation_data, country_region, province_state)]
  plt.plot(
    p_pred['Date'],
    p_pred[column],
    color=colors['very_dark_gray']
  )
  plt.plot(
    p_data['Date'],
    p_data[column],
    color=colors['light_gray'],
    marker='.',
    linestyle='None'
  )
  plt.plot(
    p_valid['Date'],
    p_valid[f'{column}Actual'],
    color=colors['light_orange'],
    marker='x',
    linestyle='None'
  )
  label_from_tick = lambda tick: pd.to_datetime(tick).strftime('%b %d')
  ticks = np.unique(np.concatenate((dates, test_dates)))[8::14]
  tick_labels = list(map(label_from_tick, ticks))
  plt.xticks(ticks, tick_labels, rotation=20, horizontalalignment='right')
  name = country_region if province_state == None else f'{country_region} {province_state}'
  name = name + f' {column}'
  plt.title(name)
  plt.show()

#%%
plot_for(subset_data, model_pred, pred_df, subset_validation_data, country_region, province_state)
plot_for('ConfirmedCases', subset_data, model_pred, pred_df, subset_validation_data, 'US', 'New York')
plot_for('Fatalities', subset_data, model_pred, pred_df, subset_validation_data, 'US', 'New York')
plot_for('ConfirmedCases', subset_data, model_pred, pred_df, subset_validation_data, 'US', 'Washington')
plot_for('Fatalities', subset_data, model_pred, pred_df, subset_validation_data, 'US', 'Washington')
plot_for('ConfirmedCases', subset_data, model_pred, pred_df, subset_validation_data, 'Italy')
plot_for('Fatalities', subset_data, model_pred, pred_df, subset_validation_data, 'Italy')
plot_for('ConfirmedCases', subset_data, model_pred, pred_df, subset_validation_data, 'Iceland')
plot_for('Fatalities', subset_data, model_pred, pred_df, subset_validation_data, 'Iceland')
plot_for('ConfirmedCases', subset_data, model_pred, pred_df, subset_validation_data, 'Germany')
plot_for('Fatalities', subset_data, model_pred, pred_df, subset_validation_data, 'Germany')
plot_for('ConfirmedCases', subset_data, model_pred, pred_df, subset_validation_data, 'Spain')
plot_for('Fatalities', subset_data, model_pred, pred_df, subset_validation_data, 'Spain')

country_region = 'China'
for province_state in subset_data[subset_criteria(subset_data, country_region)]['Province/State'].unique():
  plot_for(subset_data, model_pred, pred_df, subset_validation_data, country_region, province_state)

#%%
submission = pred_df[pred_df['ForecastId'] > -1][['ForecastId', 'ConfirmedCases', 'Fatalities']]
submission.astype('int32')
# display(submission)
submission.to_csv('submission.csv', index=False)