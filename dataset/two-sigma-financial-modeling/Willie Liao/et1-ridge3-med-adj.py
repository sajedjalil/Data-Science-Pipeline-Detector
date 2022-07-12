import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion, make_pipeline
from gc import collect

###################################
### Settings
###################################
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', '{:.4f}'.format)

N_THREADS = -1
Y_CLIP_LO, Y_CLIP_HI = -0.085, 0.085
TS_ADJ_CLIP_LO, TS_ADJ_CLIP_HI = 0.01, 3
TS_ADJ_RATIO = 0.022
CUMMED_ADJ_RATIO = 0.15
MIN_ADJ_DATA = 100
RANDOM_SEED = 20170303

cols_na = ['technical_' + str(i) for i in [0, 9, 13, 16, 20, 30, 32, 38, 44]]
cols_diff = ['technical_' + str(i) for i in [11, 13, 20, 22, 27, 30, 34, 44]] + ['derived_0']
cols_backshift = cols_diff + ['ma', 'fundamental_11']
cols_ts = ['ma', 'y_lag', 'sign_change']

env = kagglegym.make()
o = env.reset()
cols_excl = ([env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
             + [c + '_B' for c in cols_backshift] + ['ti', 'y_lag_prod', 'sign_change_sum'])
cols_orig = [c for c in o.train.columns if c not in cols_excl] + ['ma']
cols_na_count = [c + '_nan' for c in cols_orig if c not in cols_excl]

###################################
### Classes
###################################
class CountFillMissing(TransformerMixin):
    '''Create NA indicator columns, NA counts, and fill with median.'''
    def __init__(self, cols_orig, cols_na, cols_medians):
        self.cols_orig = cols_orig
        self.cols_na = cols_na
        self.cols_medians = cols_medians

    def fit(self, X=None):
        return self

    def transform(self, X):
        X['ma'] = X['technical_20'] + X['technical_13'] - X['technical_30']
        X = X.assign(nas=0, nas1=0)
        for c in self.cols_orig:
            X[c + '_nan'] = pd.isnull(X[c])
            X['nas'] += X[c + '_nan']
            if c in self.cols_na:
                X['nas1'] += X[c + '_nan']
        X.fillna(self.cols_medians, inplace=True)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class MovingAverage(TransformerMixin):
    '''Track previous values in order to calculate lags and differences.'''
    def __init__(self, cols_backshift, cols_diff, cols_medians):
        self.cols_backshift = cols_backshift
        self.cols_diff = cols_diff
        self.cols_medians = cols_medians
        self.cols_keep = list({'id', 'ma', 'y_lag', 'y_lag_prod', 'sign_change_sum', 'ti'}
                              | set(self.cols_backshift) | set(self.cols_diff))
        # Store latest features for differences and cumulative columns
        self.previous = None

    def fit(self, X=None):
        return self

    def transform(self, X):
        # Previous values
        X = pd.merge(X, self.previous, how='left', on='id', suffixes=['', '_B'], sort=False)
        for c in self.cols_backshift:
            X[c + '_B'].fillna(self.cols_medians[c], inplace=True)
            if c in self.cols_diff:
                X[c + '_D'] = X[c] - X[c + '_B']

        # Fill if no previous value
        X.ti.fillna(-1, inplace=True)
        X.loc[X.y_lag.isnull(), 'y_lag'] = X.loc[X.y_lag.isnull(), 'ma']
        X.loc[X.y_lag_prod.isnull(), 'y_lag_prod'] = X.y_lag.loc[X.y_lag_prod.isnull()] + 1.0
        X.sign_change_sum.fillna(0, inplace=True)

        # Moving Averages
        X['ti'] += 1
        X.rename(columns={'y_lag_prod': 'y_lag_prod_B', 'y_lag': 'y_lag_B'}, inplace=True)
        X['y_lag'] = 15.0 * X['ma'] - 14.0 * X['ma_B']
        X['y_lag_prod'] = X['y_lag_prod_B'] * (1.0 + X['y_lag'])
        X['y_lag_diff'] = X['y_lag_prod'] - X['y_lag_prod_B']
        X['sign_change'] = X['y_lag'] == X['y_lag_B']
        X['sign_change_sum'] += X['sign_change']
        X['sign_change_cum'] = X['sign_change_sum'] / X['ti']
        X.loc[X.ti < 10, 'sign_change_cum'] = 0.5
        X.drop(['y_lag_prod_B', 'y_lag_B'], axis=1, inplace=True)

        # Need to keep previous ids not present in current timestamp
        self.previous = pd.concat([X[self.cols_keep], self.previous.loc[~self.previous.id.isin(X.id)]])
        return X

    def fit_transform(self, X, y=None, **fit_params):
        # Previous values
        X.sort_values(['id', 'timestamp'], inplace=True)
        X.reset_index(drop=True, inplace=True)
        g = X.groupby('id')
        X['ti'] = g.cumcount()
        for c in self.cols_backshift:
            X[c + '_B'] = g[c].shift(1)
            X[c + '_B'].fillna(self.cols_medians[c], inplace=True)
            if c in self.cols_diff:
                X[c + '_D'] = X[c] - X[c + '_B']
        del g

        # Lagged target
        X['y_lag'] = 15.0 * X['ma'] - 14.0 * X['ma_B']

        # Cumulative Values
        X['y_lag_prod'] = X['y_lag'] + 1.0
        X['y_lag_prod'] = X.groupby('id')['y_lag_prod'].cumprod()
        X['y_lag_diff'] = X['y_lag_prod'] - X.groupby('id')['y_lag_prod'].shift(1)
        X['y_lag_diff'].fillna(0.0, inplace=True)

        # Sign Change
        g = X.groupby('id')['y_lag']
        X['sign_change'] = np.sign(X.y_lag) != np.sign(g.shift(1).fillna(0.0))
        g = X.groupby('id')
        X['sign_change_sum'] = g['sign_change'].cumsum()
        X['sign_change_cum'] = X['sign_change_sum'] / X['ti']
        X.loc[X.ti < 10, 'sign_change_cum'] = 0.5

        self.previous = g[self.cols_keep].last().reset_index(drop=True)
        del g
        return X


class ExtremeValues(TransformerMixin):
    '''Indicator for likely extreme values.'''
    def fit(self):
        return self

    def transform(self, X):
        X['extreme0'] = (
            (X.technical_21 < -1.6).astype(int)
            + (X.technical_35 < -1.0).astype(int)
            + (X.technical_36 < -1.0).astype(int)
            + (X.technical_21 > 2.0).astype(int)
            + (X.technical_27 < -1.3).astype(int)
            + (X.fundamental_53 < -1.0).astype(int))
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class ModelTransformer(TransformerMixin):
    '''Hack to use row and column filters in model pipeline.'''
    def __init__(self, model, cols, rows):
        self.model = model
        self.cols = cols
        self.rows = rows

    def fit(self, X, y):
        self.model.fit(X.loc[self.rows, self.cols], y.loc[self.rows])
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X.loc[:, self.cols]))


###################################
### Preprocess
###################################
print('### Preprocess')
# train = pd.read_hdf('../input/train.h5')
train = o.train
print('train before preprocess:', train.shape)
print('timestamps:', train["timestamp"].nunique())

train['ma'] = train['technical_20'] + train['technical_13'] - train['technical_30']
cols_medians = train[cols_orig].median(axis=0).to_dict()

preprocess_pipe = make_pipeline(
    CountFillMissing(cols_orig, cols_na, cols_medians)
    , MovingAverage(cols_backshift, cols_diff, cols_medians)
    , ExtremeValues()
)
train = preprocess_pipe.fit_transform(train)
print('train after preprocess:', train.shape)

print('Store previous targets for cumulative median')
y_lag_meds = train.loc[:, ['id', 'y_lag']]


###################################
### Models
###################################
cols_et = [c for c in train.columns if c not in cols_excl]
cols_lr0 = ['y_lag', 'ma', 'technical_11', 'fundamental_11', 'technical_11_B', 'fundamental_11_B']
cols_lr1 = ['y_lag', 'technical_22', 'technical_34', 'technical_22_B', 'technical_34_B']
cols_lr2 = ['ma', 'y_lag_prod', 'y_lag_diff']

post_ts10 = (train.timestamp > 10)
y_is_within_cut = (post_ts10) & (Y_CLIP_LO < train.y) & (train.y < Y_CLIP_HI)

print('MODEL: Extra Trees')
print('Features:', len(cols_et))
rfr = ExtraTreesRegressor(n_estimators=75, max_depth=8, min_samples_split=30, min_samples_leaf=16, n_jobs=N_THREADS, random_state=RANDOM_SEED)
model_et = rfr.fit(train.loc[post_ts10, cols_et], train.loc[post_ts10, 'y'])

print('Linear Regression')
model_lr0 = Ridge(fit_intercept=False)
model_lr0.fit(train.loc[y_is_within_cut, cols_lr0], train.loc[y_is_within_cut, 'y'])

model_lr1 = Ridge(fit_intercept=False)
model_lr1.fit(train.loc[y_is_within_cut, cols_lr1], train.loc[y_is_within_cut, 'y'])

model_lr2 = Ridge(fit_intercept=False)
model_lr2.fit(train.loc[y_is_within_cut, cols_lr2], train.loc[y_is_within_cut, 'y'])

models = {'et': model_et, 'lr0': model_lr0, 'lr1': model_lr1, 'lr2': model_lr2}
model_cols = {'et': cols_et, 'lr0': cols_lr0, 'lr1': cols_lr1, 'lr2': cols_lr2}
model_weights = {'et': 0.6, 'lr0': 0.22, 'lr1': 0.03, 'lr2': 0.15}


# Clean up
train.drop([c for c in train.columns if c not in ['id', 'timestamp', 'y']], axis=1, inplace=True)
del train, post_ts10, y_is_within_cut
collect()

while True:
    # Preprocess
    test = o.features    
    test = preprocess_pipe.transform(test)

    # Predict
    test['y_hat'] = 0.0
    for n, m in models.items():
        test['y_hat'] += m.predict(test[model_cols[n]]) * model_weights[n]

    # Adjust y_hat by timestamp variability
    if len(test) > MIN_ADJ_DATA:
        y_lag_med_ts = abs(test.y_lag).median()
        y_hat_med_ts = abs(test.y_hat).median()
        if y_lag_med_ts > 1e-6 and y_hat_med_ts > 1e-6:
            adj = y_lag_med_ts / y_hat_med_ts * TS_ADJ_RATIO
            adj = np.clip(adj, TS_ADJ_CLIP_LO, TS_ADJ_CLIP_HI)
            test['y_hat'] *= adj

    # Adjust y_hat by cumulative median
    y_lag_meds = pd.concat([y_lag_meds, test[['id', 'y_lag']]])
    y_lag_med = y_lag_meds.groupby('id').median().reset_index(drop=False)
    test = pd.merge(test, y_lag_med, how='left', on='id', suffixes=['', '_med'])
    test.loc[test.ti<10, 'y_lag_med'] = 0.0
    test['y_hat'] = test['y_hat'] * (1 - CUMMED_ADJ_RATIO) + test['y_lag_med'] * (CUMMED_ADJ_RATIO)

    # Clip
    test['y_hat'] = test['y_hat'].clip(Y_CLIP_LO, Y_CLIP_HI)

    # Cleanup
    pred = o.target
    pred['y'] = test['y_hat']
    test.drop([c for c in test.columns if c not in ['id', 'timestamp', 'y_hat']], axis=1, inplace=True)
    del y_lag_med
    collect()

    o, reward, done, info = env.step(pred)

    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(o.features.timestamp[0], reward, adj)
