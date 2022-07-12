# importing necessary packages
import numpy as np 
import pandas as pd 
import lightgbm as lgb
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# defining some functions
def smape(x, x_):
    """Return the smape value for two arrays."""
    return 100 * np.mean(2 * np.abs(x - x_)/(np.abs(x) + np.abs(x_)))
    
def linear_fit_slope(y):
    """Return the slope of a linear fit to a series."""
    y_pure = y.dropna()
    length = len(y_pure)
    x = np.arange(0, length)
    slope, intercept = np.polyfit(x, y_pure.values, deg=1)
    return slope

def linear_fit_intercept(y):
    """Return the intercept of a linear fit to a series."""
    y_pure = y.dropna()
    length = len(y_pure)
    x = np.arange(0, length)
    slope, intercept = np.polyfit(x, y_pure.values, deg=1)
    return intercept

# importing input datasets
train = pd.read_csv('../input/train.csv', parse_dates=True, index_col=['date'])
test = pd.read_csv('../input/test.csv', parse_dates=True, index_col=['date'])

# concatenating train and test data frames
test['sales'] = np.nan
df = pd.concat([train, test.loc[:, ['store', 'item', 'sales']]]).sort_values(by=['store', 'item'])

# adding some time-related factors
df['quarter'] = df.index.quarter
df['month'] = df.index.month
df['dow'] = df.index.weekday
df['week_block_num'] = [int(x) for x in np.floor((df.index - pd.to_datetime('2012-12-31')).days/7) + 1]
df['quarter_block_num'] = (df.index.year - 2013) * 4 + df['quarter']


# detecting and handling outliers

# finding the slope of a linear fit for sale values grouped by store, item and day of week
lin_slope_df = df.groupby(['store', 'item', 'dow'])['sales'].apply(linear_fit_slope).reset_index()
lin_slope_df.columns = ['store', 'item', 'dow', 'lin_slope']
df = df.reset_index().merge(lin_slope_df, how='left', on=['store', 'item', 'dow']).set_index('date')

# finding the intercept of a linear fit for sale values grouped by store, item and day of week
lin_intercept_df = df.groupby(['store', 'item', 'dow'])['sales'].apply(linear_fit_intercept).reset_index()
lin_intercept_df.columns = ['store', 'item', 'dow', 'lin_intercept']
df = df.reset_index().merge(lin_intercept_df, how='left', on=['store', 'item', 'dow']).set_index('date')

# fitting a linear model to the sale values grouped by store, item and day of week (trend)
df['linear_fit'] = (df['week_block_num'] - 1) * df['lin_slope'] + df['lin_intercept']

# removing the increasing trend from the sale values
df['trend_removed_sales'] = df['sales'] - df['linear_fit']

# removing the yearly seasonality from the sale values
differenced_df = df.groupby(['store', 'item', 'dow'])['trend_removed_sales'].rolling(window=53, min_periods=53).\
apply(lambda x: x[-1]-x[0]).reset_index()
differenced_df.columns = ['store', 'item', 'dow', 'date', 'diff']
differenced_df = differenced_df.sort_values(by=['store', 'item', 'date'])
df['diff'] = differenced_df['diff'].values

# normalizing the stationary sale values
diff_mean_df = df.groupby(['store', 'item', 'dow'])['diff'].mean().reset_index()
diff_mean_df.columns = ['store', 'item', 'dow', 'diff_mean']
df = df.reset_index().merge(diff_mean_df, how='left', on=['store', 'item', 'dow']).set_index('date')

diff_std_df = df.groupby(['store', 'item', 'dow'])['diff'].std().reset_index()
diff_std_df.columns = ['store', 'item', 'dow', 'diff_std']
df = df.reset_index().merge(diff_std_df, how='left', on=['store', 'item', 'dow']).set_index('date')

df['norm_diff'] = (df['diff'] - df['diff_mean']) / df['diff_std']

# identifying outliers
df['outlier'] = (abs(df['norm_diff']) > 3) * 1

# handling outliers (interpolation)
corrected_sales = []
for ind, row in df[df['outlier']==1].iterrows():
    past_week = ind - pd.Timedelta('7 days')
    next_week = ind + pd.Timedelta('7 days')
    store = row['store']
    item = row['item']
    past_week_sales = df.loc[past_week][(df.loc[past_week]['store']==store) & 
    (df.loc[past_week]['item']==item)]['sales'].values[0]
    next_week_sales = df.loc[next_week][(df.loc[next_week]['store']==store) & 
    (df.loc[next_week]['item']==item)]['sales'].values[0]
    interpolated_sales = 0.5 * (past_week_sales + next_week_sales)
    corrected_sales.append(interpolated_sales)
df.loc[df['outlier']==1, 'sales'] = corrected_sales

# removing useless columns
df.drop(columns = ['linear_fit',
                   'lin_slope', 
                   'lin_intercept', 
                   'trend_removed_sales', 
                   'diff', 
                   'diff_mean', 
                   'diff_std', 
                   'norm_diff', 
                   'outlier'], 
                   inplace=True)
                 
# sorting dataframe
df = df.sort_values(by=['item', 'store'])

# building expanding mean sale values for grouped by store, item, (day of week, month, and quarter)
# shift(1) allows to exclude the most recent sale value
for item in ['dow', 'month', 'quarter']:
    grouped_df = df.groupby(['store', 'item', item])['sales'].expanding().mean().shift(1).bfill().\
    reset_index()
    grouped_df.columns = ['store', 'item', item,'date', item + '_ex_avg_sale']
    grouped_df = grouped_df.sort_values(by=['item', 'store', 'date'])
    df[item + '_ex_avg_sale'] = grouped_df[item + '_ex_avg_sale'].values
    
# finding store,items whose mean sales value is below the 50% percentile
# (later, the prediction for year 2018 for these store,items will be multiplied by a factor smaller than one)
store_item_mean_sale_series = df.groupby(['store', 'item'])['sales'].mean()
sale_critical_value = store_item_mean_sale_series.quantile(0.5)
critical_store_item_df = store_item_mean_sale_series[store_item_mean_sale_series < sale_critical_value].\
reset_index()
critical_store_item_df.drop(columns=['sales'], inplace=True)
critical_store_item_df['critical_store_item'] = 1
df = df.reset_index().merge(critical_store_item_df, how='left', on=['store', 'item']).\
set_index('date')
df['critical_store_item'] = df['critical_store_item'].fillna(0)

# sorting dataframe
df = df.sort_values(by=['item', 'store'])

# defining useful features for lightGBM
used_feats = ['month_ex_avg_sale', 'dow_ex_avg_sale', 'quarter_ex_avg_sale',
              'month', 'dow', 'quarter_block_num', 'critical_store_item', 'sales']
df = df.loc[:, used_feats]

# defining training and testing data frames
training_df = df.loc['2013':'2017']
testing_df = df['2018']

# defining a condition for critical store, items and droping the 'critical_store_item' value
critical_store_item_mask = testing_df['critical_store_item']==1
training_df.drop(columns=['critical_store_item'], inplace=True)
testing_df.drop(columns=['critical_store_item'], inplace=True)

# developing a lightGBM model, evaluating it and investigating feature importance
X = training_df.loc[:, [col for col in training_df.columns if col not in ['sales']]]
y = training_df['sales']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# defining the parameters and hyper-parameters. For better result, the hyper-parameters should be 
# tuned carefully.
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'max_depth': 3,
    'metric' : 'mape',
    'learning_rate': 0.1
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_val,
                early_stopping_rounds=50,
                verbose_eval=2000)

preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)

print('validation smape: ', smape(y_val, preds))
print('validation mae: ', mean_absolute_error(y_val, preds))

# investigating the distribution of the error
error = y_val.values - preds

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.hist(error, EDGECOLOR='black', color='y')

# comparing the distribution of the predictin and the actual 
sm.qqplot_2samples(y_val.values, preds, line='45', ax=plt.subplot(1, 2, 2))
plt.show()

# exploring the feature importance
lgb.plot_importance(gbm, height=0.6)
plt.show()


# predicting sale values for year 2018
X_train = training_df.loc[:, [col for col in training_df.columns if col not in ['sales']]].values 
y_train = training_df['sales'].values
X_test = testing_df.loc[:, [col for col in testing_df.columns if col not in ['sales']]]
lgb_train = lgb.Dataset(X_train, y_train)

# defining the parameters and hyper-parameters. For better result, the hyper-parameters should be 
# tuned carefully.
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'max_depth': 3,
    'learning_rate': 0.1
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000)
test_preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# adding prediction to testing dataframe
testing_df.loc[:,'sales'] = test_preds

# for critical store items, the prediction is multplied by a factor slightly smaller than 1 as it 
# apears the model overpredicts them
testing_df.loc[critical_store_item_mask, 'sales'] = testing_df.loc[critical_store_item_mask, 'sales'] * 0.99

# creating submission
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['sales'] = testing_df['sales'].values
sample_submission.to_csv('submision_1.csv',index=False)
print(sample_submission.head())
