import kagglegym
import numpy as np
import pandas as pd
import math
import time
import gc   # do garbage collection manually
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class fast_BaggingRegressor(BaseEstimator):
    ''' bagging of linear regressors are still a linear model '''
    def __init__(self, base_estimator=None, n_estimators=10, random_state=123456,
                 max_samples=1.0, max_features=0, bootstrap=True, bootstrap_features=False, filter_estimators=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap  # not used yet
        self.bootstrap_features = bootstrap_features
        self.is_fitted = False
        self.model_coef = None  # np.array(n_estimators, n_features)
        self.model_constants = np.zeros((n_estimators,))  # np.array(n_estimators, 1)
        self.model_rewards = np.zeros((n_estimators, ))  # np.array(n_estimators, 1)
        self.n_features = None
        self.prng = np.random.RandomState(random_state)
        self.filter_estimators = filter_estimators
        self.max_features = max_features  # number of features instead of percentage

    def normalized_rms_error(self, y_true, y_pred):
        return np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    def fit(self, X, y):
        try:
            X = X.values
            y = y.values
        except:
            pass
        n_samples, self.n_features = X.shape
        # the first column will be used to split the classifier
        assert self.n_features >= 1, 'dimension of X should better than 1, n_samples'
        self.model_coef = np.zeros((self.n_estimators, self.n_features))

        assert self.base_estimator is not None, 'please set base_estimator'

        # implicit assumpution: base_estimator is a linear regression
        features_indices_ = np.arange(self.n_features)
        sample_indices_ = np.arange(n_samples)
        train_filter = np.zeros((n_samples, ), dtype=bool)
        current_model = clone(self.base_estimator)
        for idx_ in np.arange(self.n_estimators):
            train_filter[:] = False
            # we are doing sampling with replacement of the whole thing
            train_indices = self.prng.choice(sample_indices_, size=int(self.max_samples*n_samples), replace=True)
            train_filter[train_indices] = True

            if self.bootstrap_features:
                assert (self.max_features > 1) and (self.max_features <= self.n_features), \
                    'selected number of features should be more than 1 and not more than n_features'
                n_selected_features = int(self.max_features)
                current_features = self.prng.choice(features_indices_, size=n_selected_features, replace=False)
            else:
                current_features = features_indices_
            # inspired by sklearn estimator.predict((X[mask, :])[:, features])
            current_model.fit((X[train_filter, :])[:, current_features], y[train_filter])
            y_pred = current_model.predict((X[~train_filter, :])[:, current_features])
            # print self.rms_error(y[~train_filter], y_pred), current_model.intercept_, current_model.coef_[:]
            self.model_rewards[idx_] = self.normalized_rms_error(y[~train_filter], y_pred)
            self.model_constants[idx_] = current_model.intercept_
            self.model_coef[idx_, current_features] = current_model.coef_

            if idx_ % 50 == 0:
                print('fast_BaggingRegressor finished %d out of %d' % (idx_, self.n_estimators))

        self.final_model = clone(self.base_estimator)
        if self.filter_estimators:
            # select top performing models on out of sample
            selected_models = self.model_rewards.argsort()[-int(0.5*self.n_estimators):]
            self.final_model.coef_ = np.mean(self.model_coef[selected_models, :], axis=0)
            self.final_model.intercept_ = np.mean(self.model_constants[selected_models], axis=0)
        else:
            # linearity of the model allows for simple additions
            self.final_model.coef_ = np.mean(self.model_coef, axis=0)
            self.final_model.intercept_ = np.mean(self.model_constants, axis=0)

        self.is_fitted = True
        self.coef_ = self.final_model.coef_
        self.intercept_ = self.final_model.intercept_
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError('model is not fitted yet ')
        try:
            X = X.values
        except:
            pass
        return self.final_model.predict(X)

def _reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return(R)


def index_chunker(data_df, n_chunks):
    # returns a pd.DataFrame with number of columns depending on the chunksize
    # print('splitting data into %d chunks' % n_chunks)
    unique_timestamp = data_df["timestamp"].unique()
    n_total = len(unique_timestamp)
    chunk_size = int(n_total/n_chunks)
    chunk_boundaries = np.arange(0, n_total, chunk_size)
    index_df = pd.DataFrame(data=True, index=data_df.index, columns=range(chunk_boundaries.shape[0]))
    for idx_, pos in enumerate(chunk_boundaries):
        # append the last partial chunk to the last list
        if (pos + 2*chunk_size) > n_total:
            current_filter = (data_df.timestamp >= pos)
            index_df.ix[current_filter, idx_] = False
            if (idx_+1) <= index_df.shape[0]:
                index_df = index_df.iloc[:, :(idx_+1)]
            break
        current_filter = (data_df.timestamp >= pos) & (data_df.timestamp < (pos + chunk_size))
        index_df.ix[current_filter, idx_] = False
    return index_df

# cross validated model building to avoid stacking overfitting
# not preforming well, we need some overfitting as the signal is so weak.
def train_base_model_cv(train, y_train, model_columns, sk_model, predictor_name):
    if predictor_name not in train.columns:
        # train.loc[:, predictor_name] = np.nan --> produce SettingWithCopyWarning
        train.assign(**{predictor_name: np.nan})

    # need timestamp for chunker
    train_columns = ['timestamp']+model_columns
    X_train = train[train_columns]

    # (1) Does not respect the time flow
    # (2) some data leakage as we applied fillna on the whole training data
    n_chunks = 10
    train_test_df = index_chunker(X_train, n_chunks)
    for column_idx_ in train_test_df.columns:
        train_indices = train_test_df[column_idx_]
        sk_model.fit(X_train.ix[train_indices, model_columns], y_train[train_indices])
        train.ix[~train_indices, predictor_name] = sk_model.predict(X_train.ix[~train_indices, model_columns])

    # fit the final model
    X_train = train[model_columns]
    sk_model.fit(X_train, y_train)
    # adhoc overfitting as the signal is weak
    # train.loc[index_filter, predictor_name] = (y_pred + sk_model.predict(train.loc[index_filter, model_columns]))/2.0

def train_base_model(train, y_train, model_columns, sk_model, predictor_name):
    X_train = train[model_columns]
    sk_model.fit(X_train, y_train)
    if predictor_name not in train.columns:
        # train.loc[:, predictor_name] = np.nan --> produce SettingWithCopyWarning
        train.assign(**{predictor_name: np.nan})
    train[predictor_name] = sk_model.predict(X_train)

def process_features_training(train, id_mean_columns_list, id_mean_rolling_list):
    print('processing features during training')

    train['tech23'] = train['technical_20'] + train['technical_13'] - train['technical_30']
    train['tech23_v2'] = train['technical_20'] - train['technical_30']

    grouped_data_id = train.groupby('id')

    process_tech23 = lambda x: (x - 0.925*x.shift(1))/0.075
    approx_y = grouped_data_id['tech23'].transform(process_tech23)
    train['approx_y_prev'] = approx_y
    train['approx_y_prev'].fillna(0, inplace=True)
    train['approx_y_prev'] = train['approx_y_prev'].clip(-0.07, 0.07)

    approx_y = grouped_data_id['tech23_v2'].transform(process_tech23)
    train['approx_y_prev_v2'] = approx_y
    train['approx_y_prev_v2'].fillna(0, inplace=True)
    train['approx_y_prev_v2'] = train['approx_y_prev_v2'].clip(-0.07, 0.07)

    grouped_data_id = train.groupby('id')
    for current_column in id_mean_columns_list:
        for n_rolling_length in id_mean_rolling_list:
            if current_column == 'approx_y_prev':
                temp_data = grouped_data_id[current_column].rolling(window=n_rolling_length).median()

            else:
                temp_data = grouped_data_id[current_column].rolling(window=n_rolling_length).mean()

            # temp_data.fillna(0.0, inplace=True)
            temp_data = pd.DataFrame(temp_data.values, index=temp_data.index.get_level_values(1))
            new_column_name1 = current_column+'_id_mean_'+str(n_rolling_length)
            train[new_column_name1] = temp_data

            new_column_name2 = current_column + '_id_diff_mean_' + str(n_rolling_length)
            train[new_column_name2] = train[current_column] - train[new_column_name1]
            print('done with: %s, %s' % (new_column_name1, new_column_name2))

    grouped_data_ts = train.groupby('timestamp')
    average_values = grouped_data_ts['tech23'].agg([np.mean, np.std])
    average_values_extended = average_values.ix[train['timestamp']].reset_index(drop=True)
    train['tech23_cs_mean'] = average_values_extended['mean'].values
    train['tech23_cs_std'] = average_values_extended['std'].values

    n_rolling_length_long = 10
    average_values = grouped_data_ts['tech23'].mean().rolling(n_rolling_length_long).mean().fillna(0.0)
    train['tech23_cs_mean_10'] = average_values.ix[train['timestamp']].reset_index(drop=True).values

    train_median = train.median(axis=0)
    train = train.replace([np.inf, -np.inf], np.nan)
    train = train.fillna(train_median)
    return train, train_median

def process_features_online(test, id_mean_columns_list, id_mean_rolling_list,
                            prev_test_lists, train_median):
    print('processing features during prediction')

    max_rolling_length = np.max(id_mean_rolling_list)

    test['tech23'] = test['technical_20'] + test['technical_13'] - test['technical_30']
    test['tech23_cs_mean'] = test['tech23'].mean(axis=0)
    test['tech23_cs_std'] = test['tech23'].std(axis=0)
    # n_ids = test.shape[0]
    test_prev = prev_test_lists[-1]
    # here is should be setting by common id
    approx_y_prev_values = test[['id', 'tech23']].set_index('id') - 0.925*(test_prev[['id', 'tech23']].set_index('id'))
    approx_y_prev_values /= 0.075

    test['approx_y_prev'] = approx_y_prev_values.loc[test.id].values
    test['approx_y_prev'].fillna(0, inplace=True)

    test['tech23_v2'] = test['technical_20'] - test['technical_30']
    approx_y_prev_values = test[['id', 'tech23_v2']].set_index('id') - 0.925*(test_prev[['id', 'tech23_v2']].set_index('id'))
    approx_y_prev_values /= 0.075

    test['approx_y_prev_v2'] = approx_y_prev_values.loc[test.id].values
    test['approx_y_prev_v2'].fillna(0, inplace=True)

    # test['approx_y_prev_cs_mean'] = test['approx_y_prev'].mean()

    prev_test_lists.append(test)
    if len(prev_test_lists) > max_rolling_length:
        prev_test_lists = prev_test_lists[1:]

    test_temp = test.set_index(test.id)
    for n_rolling_length in id_mean_rolling_list:
            test_grouped_data_id = pd.concat(prev_test_lists[-n_rolling_length:], axis=0).groupby('id')
            for current_column in id_mean_columns_list:
                if current_column == 'approx_y_prev':
                    temp_data = test_grouped_data_id[current_column].median()
                else:
                    temp_data = test_grouped_data_id[current_column].mean()

                new_column_name1 = current_column+'_id_mean_'+str(n_rolling_length)
                test_temp[new_column_name1] = temp_data

                new_column_name2 = current_column + '_id_diff_mean_' + str(n_rolling_length)
                test_temp[new_column_name2] = test_temp[current_column] - test_temp[new_column_name1]
                # print('done with: %s, %s' % (new_column_name1, new_column_name2))

    test_grouped_data_id = pd.concat(prev_test_lists, axis=0).groupby('id')
    test_temp['tech23_cs_mean_10'] = test_grouped_data_id['tech23_cs_mean'].mean()

    # assert len(test_lists) == 10
    # print(test.timestamp[0], approx_y_prev_cs_mean_smoothed[0])

    test = test_temp.reset_index(drop=True)
    test = test.fillna(train_median)
    return test, prev_test_lists


env = kagglegym.make()
observation = env.reset()
# excl = [env.ID_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]  # env.SAMPLE_COL_NAME
# col = [c for c in observation.train.columns if c not in excl]

train = observation.train
# train = pd.read_hdf('../input/train.h5')

id_mean_columns_list = ['tech23', 'approx_y_prev', 'tech23_v2', 'approx_y_prev_v2']
id_mean_rolling_list = [3, 5, 7, 10]
train, train_median = process_features_training(train, id_mean_columns_list, id_mean_rolling_list)

# https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
# Observed with histograns:
low_y_cut = -0.085
high_y_cut = 0.092

mean_normalization_column = 'tech23_cs_mean_10'
volatility_column = 'tech23_cs_std'
cs_factor = 0.25  # try to adjust the mean y_train + some trial and error

print('mean_normalization_columns: %s, current cs_factor: %f' % (mean_normalization_column, cs_factor))

y_train = train['y'] - cs_factor*train[mean_normalization_column]
y_train /= train[volatility_column]

base_models = []
# simple_model = LinearRegression(fit_intercept=True)
simple_model = make_pipeline(StandardScaler(), LinearRegression(fit_intercept=True))

# now one could add tons of models here
base_models = []
base_models_columns = [['tech23', 'tech23_id_mean_3'],
                       ['technical_40', 'fundamental_11'],
                       ['approx_y_prev', 'approx_y_prev_id_mean_3', 'approx_y_prev_id_mean_10'],
                       ['approx_y_prev_id_mean_3', 'tech23_id_diff_mean_10'],
                       ['approx_y_prev_id_mean_3', 'approx_y_prev_id_mean_7', 'fundamental_11'],
                       ['tech23_id_diff_mean_3', 'fundamental_11'],
                       ['tech23', 'tech23_id_mean_5'],
                       ['fundamental_11', 'approx_y_prev_id_mean_7'],
                       ['approx_y_prev_id_mean_5', 'approx_y_prev_id_mean_10'],
                       ['technical_19', 'tech23_v2_id_diff_mean_10'],
                       ['approx_y_prev', 'approx_y_prev_id_mean_3', 'approx_y_prev_id_mean_5'],
                       ]

n_base_models = len(base_models_columns)
print('start of training')
t0 = time.time()

collected = gc.collect()
print('Garbage collector: collected %d objects.' % collected)

for idx_ in np.arange(n_base_models):
    model_columns_ = base_models_columns[idx_]
    predictor_name = 'pred_%d' % idx_
    sk_model_ = clone(simple_model)
    train_base_model(train, y_train, model_columns_, sk_model_, predictor_name)
    # print('train_base_model %d, with parameters intercept_ and coef_: %s, %s' % (idx_, sk_model_.intercept_, sk_model_.coef_))
    print('train_base_model %d' % (idx_))
    base_models.append(sk_model_)

# using all data in model stacking -- overfitting
base_models_outputs = ['pred_%d' % idx_ for idx_ in np.arange(n_base_models)]
model_final_columns = base_models_outputs  # + ['fundamental_11', 'approx_y_prev', 'approx_y_prev_id_mean_3', 'approx_y_prev_id_mean_10']

X_train = train[model_final_columns]
predictor_stacked_name = 'stacked_pred'

# clone mess-up things with random state
# More features is somehow good -- max_features 0.6 is better and 0.3 for 7 features
sk_model_final = fast_BaggingRegressor(LinearRegression(fit_intercept=True), n_estimators=3000,
                                       random_state=565776, max_samples=1.0, max_features=3,
                                       bootstrap=True, bootstrap_features=True, filter_estimators=True)

sk_model_final.fit(X_train, y_train)
print('sk_model_final with parameters intercept_ and coef_: %s, %s' % (sk_model_final.intercept_, sk_model_final.coef_))

print('Done training in %f sec' % (time.time()-t0))

# this is for generating features online
max_rolling_length = np.max(id_mean_rolling_list)
prev_test_lists = list(train.groupby('timestamp'))[-max_rolling_length:]
prev_test_lists = [x[1] for x in prev_test_lists]

collected = gc.collect()
print('Garbage collector: collected %d objects.' % collected)

i = 0
reward_ = []
while True:
    test = observation.features
    pred = observation.target

    test, prev_test_lists = process_features_online(test, id_mean_columns_list, id_mean_rolling_list,
                                                    prev_test_lists, train_median)

    for temp_pred_name in model_final_columns + [predictor_stacked_name]:
        if temp_pred_name not in test.columns:
            test[temp_pred_name] = np.nan

    for idx_local_ in np.arange(n_base_models):
        model_columns_ = base_models_columns[idx_local_]
        predictor_name = 'pred_%d' % idx_local_
        sk_model_ = base_models[idx_local_]
        X_current = test[model_columns_]
        test[predictor_name] = sk_model_.predict(X_current)

    # predict on the stacked model
    X_current = test[model_final_columns]
    test[predictor_stacked_name] = sk_model_final.predict(X_current)

    test[predictor_stacked_name].replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

    test['y_pred'] = test[predictor_stacked_name]
    test['y_pred'] *= test[volatility_column]
    test['y_pred'] += cs_factor*test[mean_normalization_column]

    test['y_pred'] = test['y_pred'].clip(low_y_cut, high_y_cut)

    pred['y'] = test.loc[:, 'y_pred']
    observation, reward, done, info = env.step(pred[['id', 'y']])
    reward_.append(reward)
    if i % 100 == 0:
        print(reward, np.mean(np.array(reward_)))
        collected = gc.collect()
        print('Garbage collector: collected %d objects.' % collected)

    i += 1
    if done:
        print("finished ...", info["public_score"])
        break
