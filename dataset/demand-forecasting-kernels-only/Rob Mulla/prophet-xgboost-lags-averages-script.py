import pandas as pd
import numpy as np
import xgboost as xgb
import datetime as dt
from fbprophet import Prophet
from tqdm import tqdm, tqdm_notebook
import logging
import warnings
warnings.filterwarnings('ignore')

#################
# Setup logging
#################

logger = logging.getLogger('kaggle_demand_forecasting')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('./kaggle_demand.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
# Supress fbprophet output
logging.getLogger('fbprophet').setLevel(logging.WARNING)

####################
# HELPER FUNCTIONS
####################


def create_prophet_features(train):
    """
    Train a seperate fbprophet model on each item/store.
    Use the results as features used by the final model
    """
    logger.info('creating prophet models for features')
    grouped = train.groupby(['item', 'store'])
    prophet_results = []
    for i, d in tqdm(grouped):
        m = Prophet(uncertainty_samples=300, daily_seasonality=False)
        m.fit(d.rename(columns={'date': 'ds', 'sales': 'y'}))
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        forecast['item'] = i[0]
        forecast['store'] = i[1]
        prophet_results.append(forecast)
    prophet_features = pd.concat(prophet_results)
    prophet_features = prophet_features.rename(columns={'ds': 'date'})
    prophet_features.to_csv('prophet_results.csv')
    return prophet_features


def add_sametime_last_year(df, date_col='date',
                           by_vars=['store', 'item'], target='sales'):
    """
    Using the ``date_col`` adds columns with values for ``target``
    on the same date the prior year for the same ``by_vars``.

    Parameters
    ---------
    df : pandas dataframe
         Dataframe with ``date_col``, ``by_vars`` and ``target`` that
         will be used to identify the same date last year.
    date_col : string of date column
    by_vars : list of by vars
    target : string of target column
    Returns
    ---------
    df : pandas dataframe
        Dataframe with same time last year columns added.
    """
    logger.info('Adding lag features')

    days_back_list = [x * 7 for x in range(13, 51)] + [357, 364, 371,
                                                       721, 728, 735,
                                                       1085, 1092, 1099]

    df = df[[date_col] + by_vars + [target]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    new_cols = []
    for days_back in days_back_list:
        df_for_join = df.copy()
        df_for_join['{}_min{}'.format(date_col, days_back)] = \
            df_for_join[date_col] - pd.Timedelta('-{} days'.format(days_back))
        left_on = [date_col] + by_vars
        right_on = ['{}_min{}'.format(date_col, days_back)] + by_vars
        df_merged = df.merge(df_for_join,
                             left_on=left_on,
                             right_on=right_on,
                             suffixes=('', '_min{}'.format(days_back)),
                             how='left') \
            .drop(columns=['{}_min{}'.format(date_col, days_back)])
        new_cols.append(df_merged['{}_min{}'.format(target, days_back)])
    new_cols_df = pd.DataFrame(new_cols).T
    df_with_lags = pd.concat([df, new_cols_df], axis=1)

    df_with_lags_avgs = add_averages(df_with_lags)
    return df_with_lags_avgs


def add_averages(df):
    logger.info('Adding averages features')

    df['month'] = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.weekofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['daily_avg'] = df.groupby(['item', 'store', 'dayofweek'])[
        'sales'].transform('mean')
    df['monthly_avg'] = df.groupby(['item', 'store', 'month'])[
        'sales'].transform('mean')
    df['week_avg'] = df.groupby(['item', 'store', 'weekofyear'])[
        'sales'].transform('mean')
    df['monthly_store_avg'] = df.groupby(['store', 'month'])[
        'sales'].transform('mean')
    df['monthly_item_avg'] = df.groupby(['item', 'month'])[
        'sales'].transform('mean')
    df['store_item_avg'] = df.groupby(['store', 'item'])[
        'sales'].transform('mean')
    return df.drop(['month', 'dayofweek'], axis=1)

# Create Features


def create_features(df, labeled=True):
    """
    Creates the features used by model from raw data
    """
    logger.info('Creating features used by model')

    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    # Sine waves representation of doy, completely out of shift from eachother
    df['cos_doy'] = np.cos((df['dayofyear'] * 2 * np.pi) / 364)
    df['neg_sin_doy'] = - np.sin((df['dayofyear'] * 2 * np.pi) / 364)

    # Make Dummies
    store_dummies = pd.get_dummies(df['store'], prefix='store')
    items_dummies = pd.get_dummies(df['item'], prefix='item')
    dow_dummies = pd.get_dummies(df['dayofweek'], prefix='dow')

    # Lag variables
    lags = df[[x for x in df.columns if x[:6] == 'sales_']]

    df.loc[df['store'].isin([2, 8]) & df['month'].isin(
        [6, 7]), 'problem_child'] = 1
    df['problem_child'] = df['problem_child'].fillna(0)

    df.loc[df['store'].isin([2, 8]), 'problem_stores'] = 1
    df['problem_stores'] = df['problem_stores'].fillna(0)

    df.loc[(df['dayofyear'] > 150) & (df['dayofyear'] < 250), 'is_summer'] = 1
    df['is_summer'] = df['is_summer'].fillna(0)

    all_features = [df['dayofweek'], df['quarter'], df['month'],
                    df['year'], df['dayofyear'], df['dayofmonth'],
                    df['weekofyear'], df['cos_doy'], df['neg_sin_doy'],
                    store_dummies, items_dummies, dow_dummies,
                    lags, df['store'], df['item'],
                    df['daily_avg'], df['monthly_avg'], df['problem_child'],
                    df['store_item_avg'], df['problem_stores'],
                    df['is_summer'],
                    df['week_avg'], df['monthly_store_avg'],
                    df['monthly_item_avg'],
                    df['trend'], df['yhat_lower'], df['yhat_upper'],
                    df['trend_lower'], df['trend_upper'], df['additive_terms'],
                    df['additive_terms_lower'], df['additive_terms_upper'],
                    df['weekly'], df['weekly_lower'], df['weekly_upper'],
                    df['yearly'], df['yearly_lower'], df['yearly_upper'],
                    df['yhat']]

    X = pd.concat(all_features, axis=1)

    if labeled:
        y = df['sales']
        return X, y, np.log1p(y)
    return X


def filter_features(X, features):
    """
    Filters X down to just the features we want the model to use.
    Params
    ------
    X (Pandas dataframe) : Dataframe with all features.
    features (list) : list of features to keep columns
    """
    logger.info('filtering down features')
    return X[features]

####################
# PIPELINE
####################

def preprocess(train, test, load_prohet_csv=False):
    """
    Preprocess train and test including.
    1. Joining train/test
    2. Creating prophet features
    3. Creating lags and mean values
    4. Splitting back into train/test
    """
    logger.info('preprocessing data')

    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])

    all_data = pd.concat([train, test.drop('id', axis=1)], sort=False).copy()
    all_data = all_data.reset_index().drop('index', axis=1)
    all_data['date'] = pd.to_datetime(all_data['date'])

    # Add prophet features and merge on all data
    if load_prohet_csv:
        prophet_features = pd.read_csv('prophet_results.csv', index_col=[0])
        prophet_features['date'] = pd.to_datetime(prophet_features['date'])
    else:
        prophet_features = create_prophet_features(train)

    # Prophet feature columns
    all_data = all_data.merge(prophet_features, on=[
                              'date', 'item', 'store'], how='left')

    # Create lags and add prophet features
    all_data_lags = add_sametime_last_year(all_data)

    all_data_lags = all_data_lags.merge(
        prophet_features, on=['date', 'item', 'store'], how='left')

    # Adding lags
    test_with_lags = all_data_lags.loc[all_data_lags['sales'].isna()].copy()
    train_with_lags = all_data_lags.loc[~all_data_lags['sales'].isna()].copy()

    logger.info('test shape: ', test.shape)
    logger.info('train shape: ', train.shape)
    logger.info('test_with_logs shape:', test_with_lags.shape)
    logger.info('train_with_logs shape:', train_with_lags.shape)

    logger.info('train_with_lags columns:')
    logger.info([col for col in train_with_lags.columns])

    # Create Features
    X_train, y_train, y_log_train = create_features(train_with_lags)
    X_test = create_features(test_with_lags, labeled=False)
    return X_train, y_log_train, X_test


def run_pipeline_final(train, X_train, y_log_train,
                       features, params, verbose=True):
    """
    Pipeline function that runs the model process based on input data
    and features.
    """
    logger.info('running the pipeline')

    # Filter Features
    X_train = filter_features(X_train, features)

    # Give more weight to the month's we will submit for in the end
    train.loc[train.date.dt.month.isin([1, 2, 3]), 'sample_weight'] = 1
    train['sample_weight'] = train['sample_weight'].fillna(0.2)

    reg = xgb.XGBRegressor(n_jobs=4, **params)

    reg.fit(X=X_train, y=y_log_train,
            eval_set=[(X_train, y_log_train)],
            eval_metric='rmse',
            early_stopping_rounds=15,
            verbose=verbose,
            sample_weight=train['sample_weight'])

    return reg


def create_df_with_preds(reg, test, train,
                         X_train, X_test, features,
                         save_csv=True):
    """
    Creates test and train with the predictions added
    """
    logger.info('creating dataframe with predictions')

    X_test = filter_features(X_test, features)
    X_train = filter_features(X_train, features)

    test['sales_pred_test'] = np.expm1(reg.predict(X_test))
    train['sales_pred_train'] = np.expm1(reg.predict(X_train))

    joined_preds = pd.concat([test, train])

    now = dt.datetime.now()
    dateandtime = now.strftime('%b%d-%H%m')
    joined_preds.to_csv('joinedpreds-{}.csv'.format(dateandtime), index=False)
    return joined_preds


def round_to_half(number):
    """Round a number to the closest half integer.
    >>> round_of_rating(1.3)
    1.5
    >>> round_of_rating(2.6)
    2.5
    >>> round_of_rating(3.0)
    3.0
    >>> round_of_rating(4.1)
    4.0"""

    return round(number * 2) / 2


def create_submit(reg, X_test, submit, features):
    """
    Creates the submission CSV
    """
    logger.info('creating submission csv')

    X_test = filter_features(X_test, features)
    submit['sales'] = np.expm1(reg.predict(X_test))
    now = dt.datetime.now()
    dateandtime = now.strftime('%b%d-%H%m')
    submit.to_csv('submit-{}.csv'.format(dateandtime), index=False)

    # Half Rounded
    submit_halfround = submit.copy()
    submit_halfround['sales'] = submit_halfround['sales'].apply(round_to_half)
    submit_halfround.to_csv(
        'submit_halfround-{}.csv'.format(dateandtime), index=False)

    # Magic Number?!?
    magic_number = 0.981
    submit_magic = submit.copy()
    submit_magic['sales'] = submit_magic['sales'] * magic_number
    submit_magic.to_csv('submit_magic-{}.csv'.format(dateandtime), index=False)
    return

####################
# DO STUFF #########
####################


# Read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submit = pd.read_csv('../input/sample_submission.csv')

# 1. Preprocess
try:
    X_train, y_log_train, X_test = preprocess(train, test,
                                              load_prohet_csv=False)
except Exception as e:
    logger.error('Preprocessing broke.')
    logger.error(e)


# 2. Set Features and params to use
params = {'subsample': 0.8,
          'n_estimators': 50000,
          'min_child_weight': 5,
          'max_depth': 3,
          'learning_rate': 0.001,
          'gamma': 0.5,
          'colsample_bytree': 0.6}

features = ['dayofweek', 
            'quarter',
            'month', 'year',
            'dayofmonth',
            'weekofyear',
            'cos_doy',
            'neg_sin_doy',
            'dayofyear', 'store', 'item',
        #     'sales_min91', 'sales_min98',
        #     'sales_min105', 'sales_min112', 'sales_min119',
        #     'sales_min126', 'sales_min133', 'sales_min140', 'sales_min147',
        #     'sales_min154', 'sales_min161', 'sales_min168',
        #     'sales_min175', 'sales_min182', 'sales_min189', 'sales_min196',
        #     'sales_min203', 'sales_min210', 'sales_min217',
        #     'sales_min224', 'sales_min231', 'sales_min238', 'sales_min245',
        #     'sales_min252', 'sales_min259', 'sales_min266',
        #     'sales_min273', 'sales_min280', 'sales_min287', 'sales_min294',
        #     'sales_min301', 'sales_min308', 'sales_min315',
        #     'sales_min322', 'sales_min329', 'sales_min336', 'sales_min343',
        #     'sales_min350', 'sales_min357', 'sales_min364',
        #     'sales_min371', 'sales_min721', 'sales_min728', 'sales_min735',
        #     'sales_min1085', 'sales_min1092', 'sales_min1099',
        #     'daily_avg', 'monthly_avg', 'store_item_avg', 'problem_stores',
        #     'is_summer', 'week_avg', 'monthly_store_avg', 'monthly_item_avg',
            'trend', 'yhat', 'yhat_lower', 'yhat_upper',
            'trend_lower', 'trend_upper', 'additive_terms',
            'additive_terms_lower', 'additive_terms_upper',
            'weekly', 'weekly_lower', 'weekly_upper',
            'yearly', 'yearly_lower', 'yearly_upper']

# 3. Train Model
reg = run_pipeline_final(train, X_train, y_log_train,
                         features, params, verbose=True)

bst = reg.get_booster()
logger.info('FSCORES:')
logger.info(bst.get_fscore())

# 4. Save Predictions for later evaluation
try:
    _ = create_df_with_preds(reg, test, train, X_train,
                             X_test, features, save_csv=True)
except Exception as e:
    logger.error('Creating combined df broke.')
    logger.error(e)

# 5. Create Submission
try:
    create_submit(reg, X_test, submit, features)
except Exception as e:
    logger.error('creating submission broke.')
    logger.error(e)