import kagglegym
import numpy as np
import pandas as pd
import xgboost
import gc
import numpy.random
import sklearn.linear_model
from collections import defaultdict
psrng = numpy.random.RandomState()

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

gc.collect()
    

drop_cols = [
    colname
    for colname in observation.train.columns
    if colname not in ['id', 'timestamp', 'technical_20', 'technical_30', 'fundamental_11', 'fundamental_53', 'technical_27', 'y', 'count_null', 'proxy_y_ma']
    ]

observation.train['count_null'] = observation.train.isnull().sum(axis=1)
observation.train['proxy_y_ma'] = observation.train['technical_20'] - observation.train['technical_30']
observation.train = observation.train.drop(drop_cols, axis=1)
means = observation.train.drop(['id', 'y'], axis=1).groupby('timestamp').mean()
stddevs = observation.train.drop(['id', 'y'], axis=1).groupby('timestamp').std()
observation.train = pd.merge(
        left=observation.train,
        right=means,
        left_on='timestamp',
        right_index=True,
        how='left',
        suffixes=['', '_mean']
    )

observation.train = pd.merge(
        left=observation.train,
        right=stddevs,
        left_on='timestamp',
        right_index=True,
        how='left',
        suffixes=['', '_stddev']
    )
lags = [1, 5, 10, 25]
timestamp_lag_cols = [
    'timestamp_lag_' + str(lag)
    for lag in lags
    ]

for lag in lags:
    observation.train['timestamp_lag_' + str(lag)] = observation.train['timestamp'] - lag
    
orig_cols = [colname for colname in observation.train.columns]

def lag_merger(
        left_df,
        right_df,
        lag,
        right_keep_cols,
    ):
    return pd.merge(
        left=left_df,
        right=right_df[right_keep_cols].rename(columns={'timestamp':'timestamp_lag'}).drop(['y'] + timestamp_lag_cols, 1),
        left_on=['id', 'timestamp_lag_' + str(lag),],
        right_on=['id', 'timestamp_lag'],
        how='left',
        suffixes=('', '_lag_' + str(lag)),
    ).drop('timestamp_lag', 1)

for i_lag, lag in enumerate(lags):
    observation.train = lag_merger(observation.train, observation.train, lag, orig_cols)

for column in observation.train:
    if (
            column.startswith('derived_')
            or column.startswith('fundamental_')
            or column.startswith('technical_')
            or column.startswith('proxy_y_ma')
            or column == 'count_null'
        ) and (
            'lag' not in column
        ):
        for lag in lags:
            observation.train['diff_{}_lag_{}'.format(column, lag)] = np.where(
                (pd.isnull(observation.train['{}_lag_{}'.format(column, lag)])) | (pd.isnull(observation.train[column])),
                np.nan,
                observation.train[column] - observation.train['{}_lag_{}'.format(column, lag)],
            )
            
drop_cols = [
    colname
    for colname in observation.train.columns
    if
        'lag' in colname
        and not colname.startswith('diff')
        and not colname.startswith('timestamp')
        and not colname == 'proxy_y'
    ]
    
observation.train['proxy_y'] = (observation.train['proxy_y_ma'] - 0.92* observation.train['proxy_y_ma_lag_1']) / 0.07
observation.train = observation.train.drop(drop_cols, axis=1)

for column in observation.train.columns:
    print(column)

gc.collect()

train_features=[
    colname
    for colname in observation.train
    if
        not colname in ('id', 'y')
        and not colname.startswith('timestamp')
        and not 'proxy_y' in colname
]
print(train_features)

print('Creating training DMatrix')
train_dmatrix = xgboost.DMatrix(
    observation.train.fillna(-9999)[train_features],
    label=observation.train['y'],
    feature_names=train_features,  
)

params = {
        'eta': 0.01,
        'max_depth': 8,
        'objective': 'reg:linear',
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'base_score': 0.0,
    }

max_trees = 200
print('Fitting XGB model')
xgb_fit = xgboost.train(
    params=params,
    dtrain=train_dmatrix,
    num_boost_round=max_trees,
    verbose_eval=5,
)

max_lag = max(lags)
print(max_lag)
max_timestamp = observation.train['timestamp'].max()
print(max_timestamp)

observation.train = observation.train[observation.train['timestamp'] > (max_timestamp - max_lag - 5)]
gc.collect()
print(len(observation.train))

train = observation.train

def lag_merger_feature(
        left_df,
        right_df,
        lag,
        right_keep_cols,
    ):
    return pd.merge(
        left=left_df,
        right=right_df[right_keep_cols].rename(columns={'timestamp':'timestamp_lag'}).drop(timestamp_lag_cols, 1),
        left_on=['id', 'timestamp_lag_' + str(lag),],
        right_on=['id', 'timestamp_lag'],
        how='left',
        suffixes=('', '_lag_' + str(lag)),
    ).drop('timestamp_lag', 1)

all_used_trees = list()
point_reward_list = list()
proxy_y_list = list()
count_observation = 0
while True:

    count_observation = count_observation + 1
    timestamp = observation.features["timestamp"][0]
    
    drop_cols = [
        colname
        for colname in observation.features
        if colname not in ['id', 'timestamp', 'technical_20', 'technical_30', 'fundamental_11', 'fundamental_53', 'technical_27', 'count_null', 'proxy_y_ma']
        ]

    observation.features['count_null'] = observation.features.isnull().sum(axis=1)
    observation.features['proxy_y_ma'] = observation.features['technical_20'] - observation.features['technical_30']
    observation.features = observation.features.drop(drop_cols, axis=1)
    means = observation.features.drop(['id'], axis=1).groupby('timestamp').mean()
    stddevs = observation.features.drop(['id'], axis=1).groupby('timestamp').std()
    observation.features = pd.merge(
            left=observation.features,
            right=means,
            left_on='timestamp',
            right_index=True,
            how='left',
            suffixes=['', '_mean']
        )

    observation.features = pd.merge(
            left=observation.features,
            right=stddevs,
            left_on='timestamp',
            right_index=True,
            how='left',
            suffixes=['', '_stddev']
        )

    if timestamp % 25 == 0:
        print("Timestamp #{}".format(timestamp))
    for lag in lags:
        observation.features['timestamp_lag_' + str(lag)] = observation.features['timestamp'] - lag
    if count_observation == 1:
        cols_to_keep = [colname for colname in observation.features.columns]

    for i_lag, lag in enumerate(lags):
        if i_lag == 0:
            left_df = observation.features
        else:
            left_df = feature_munge

        if count_observation <= lag:
            feature_munge = lag_merger_feature(left_df, train, lag, cols_to_keep)
        else:
            feature_munge = lag_merger_feature(left_df, feature_stack, lag, cols_to_keep)

    feature_munge['proxy_y'] = (feature_munge['proxy_y_ma'] - 0.92* feature_munge['proxy_y_ma_lag_1']) / 0.07
    lag_1_proxy_y = float(feature_munge[['proxy_y']].mean())
    if count_observation > 1:
        proxy_y_list.append(lag_1_proxy_y)
    for column in feature_munge.columns:
        if (
                column.startswith('derived_')
                or column.startswith('fundamental_')
                or column.startswith('technical_')
                or column.startswith('proxy_y_ma')
                or column == 'count_null'
            ) and (
                'lag' not in column
            ):
            for lag in lags:
                feature_munge['diff_{}_lag_{}'.format(column, lag)] = np.where(
                    (pd.isnull(feature_munge['{}_lag_{}'.format(column, lag)])) | (pd.isnull(feature_munge[column])),
                    np.nan,
                    feature_munge[column] - feature_munge['{}_lag_{}'.format(column, lag)],
                )

    if count_observation == 1:
        feature_stack = feature_munge
    else:
        if count_observation % 25 == 0:
            feature_stack = pd.concat([
                feature_stack[feature_stack['timestamp'] > (timestamp - max_lag - 5)], 
                feature_munge,
            ])
            gc.collect()
        else:
            feature_stack = pd.concat([
                feature_stack, 
                feature_munge,
            ])

    pred_dmatrix = xgboost.DMatrix(
        feature_munge.fillna(-9999)[train_features],
        feature_names=train_features,  
    )
    if count_observation <= 25:
        best_tree_guess = int(max(
            min(
                psrng.beta(2,2)*max_trees,
                max_trees,
                ),
            1,
            ))
    else:
        rand_sample = (psrng.beta(2,2,20)*max_trees).clip(1,max_trees)
        weights = [
            0.9925 ** x
            for x in reversed(range(len(all_used_trees)))
            ]
        lin_model = sklearn.linear_model.LinearRegression()
        fit_model = lin_model.fit(
            np.array([all_used_trees,
               [x**2 for x in all_used_trees]
               ]).transpose(),
            np.array(point_reward_list, ndmin=2).transpose(),
            sample_weight=np.array(weights),
            )
        rand_tree_fit = fit_model.predict(
            np.array([rand_sample,
               [x**2 for x in rand_sample]
               ]).transpose()
            )
        best_tree_guess = int(rand_sample[rand_tree_fit.argmax()])
                                                
    all_used_trees.append(best_tree_guess)
    observation.target.y = xgb_fit.predict(pred_dmatrix, ntree_limit=best_tree_guess)

    observation, reward, done, info = env.step(observation.target)
    point_reward_list.append(reward)
    if done:
        break
    
print(info)