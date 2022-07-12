#
# Utility Notebook for Jane St Competition
#

# Imports
# Numerical
import pandas as pd
import numpy as np
# Plots
# import seaborn as sns
# import matplotlib.pyplot as plt
# Other
import warnings
warnings.filterwarnings('ignore')


class Utility:

    def __init__(self, folder_competition):
        self.folder = folder_competition
        self.features = pd.read_csv(
            f'{self.folder}/features.csv'
            ).set_index('feature')

    def filepath_train(self):
        return f'{self.folder}/train.csv'

    def select_tags(self, query, pretty=True):
        tags = self.features.query(query)
        if not pretty:
            return tags
        tags_summary_tags = tags.sum(axis=0)
        non_zero_tags = tags_summary_tags[
            tags_summary_tags != 0
        ].index.tolist()
        tags_summary_features = tags[non_zero_tags].sum(axis=1)
        non_zero_features = tags_summary_features[
            tags_summary_features != 0
        ].index.tolist()
        return tags.loc[non_zero_features, non_zero_tags]

    def get_features(self, query):
        tags = self.select_tags(query, pretty=True)
        return tags.index.tolist()

    @staticmethod
    def add_intraday_ts(train):
        # train: raw trained data from the competition
        # 'date', 'ts_id', necessary
        # add another column intraday_ts of value between [0, 1]
        # Proxy for "time" variable
        start_of_days = train.groupby('date').agg({'ts_id': 'min'})
        end_of_days = train.groupby('date').agg({'ts_id': 'max'})
        train['intraday_ts'] = 0
        for day in train['date'].unique():
            indexes_day = train[train['date'] == day].index
            ts_id_sod = start_of_days.loc[day].values[0]
            ts_id_eod = end_of_days.loc[day].values[0]
            train.loc[indexes_day, 'intraday_ts'] = \
                (train.loc[indexes_day, 'ts_id'] - ts_id_sod) \
                / (ts_id_eod - ts_id_sod)
        return train

    def build_train_na(self, train, features=None):
        if not features:
            features = self.get_features('tag_0 == tag_0')
        train_na = train[['date', 'ts_id', 'intraday_ts'] + features]
        train_na[features] = train_na[features].isna().astype(int)
        return train_na

    def add_stock_id(self, train):
        # feature_41, 42, 43 are necessary to perform this operation
        embedding_stocks = train[
            ['feature_41', 'feature_42', 'feature_43']
        ].drop_duplicates()
        embedding_stocks['stock_id'] = np.arange(embedding_stocks.shape[0])
        embedding_stocks['stock_id'] = embedding_stocks['stock_id'].astype(str)

        stocks = pd.merge(
            train,
            embedding_stocks,
            on=['feature_41', 'feature_42', 'feature_43']
        )
        return stocks

    def add_stock_id_all(self, train):
        new_train = []
        for day in train['date'].unique():
            train_date = train[train['date'] == day]
            new_train.append(
                self.add_stock_id(train_date)
            )
        return pd.concat(new_train)

    def add_feature_0(self, train, cols=None):
        if not cols:
            cols = self.get_features('tag_6') + [
                'resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4'
            ]
        for f in cols:
            train[f'{f}_x0'] = train[f] * train['feature_0']
        return train

    def get_n_trades(self, train):
        n_trades = train.groupby(['date', 'stock_id'], as_index=False).agg(
            {'ts_id': 'count'}
        ).rename(
            columns={'ts_id': 'n_trades'}
        ).sort_values('n_trades', ascending=False)
        return n_trades


class Curve:
    def __init__(self, epsilon, alpha_prior, t, x):
        self.alpha_tilde = alpha_prior
        self.alpha = alpha_prior
        self.epsilon = epsilon
        self.times = [t]
        self.points = [x]
        self.last_time = t
        self.last_point = x

    def add_point(self, t, x):
        delta_t = t - self.last_time
        delta_x = x - self.last_point
        new_alpha = delta_x / delta_t
        new_alpha_tilde = \
            self.epsilon * delta_t * new_alpha \
            + (1 - self.epsilon * delta_t) * self.alpha_tilde
        self.times.append(t)
        self.points.append(x)
        self.last_time = t
        self.last_point = x
        self.alpha = new_alpha
        self.alpha_tilde = new_alpha_tilde

    def predict(self, t):
        delta_t = t - self.last_time
        return self.last_point + delta_t * self.alpha_tilde

    def get_distance(self, t, x):
        prediction = self.predict(t)
        return ((x - prediction) ** 2).sum()


class Cluster1DSmoothFunctions:

    def __init__(
        self,
        distance_threshold,
        t_threshold,
        epsilon,
        verbose=False
    ):
        self.distance_threshold = distance_threshold
        self.t_threshold = t_threshold
        self.epsilon = epsilon
        self.verbose = verbose

    def run(self, points, alpha_prior):
        curves = []
        times = []
        n_points_seen = 0
        if self.verbose:
            print('Going through all points...')
        while points:
            t, x = points.pop(0)
            times.append(t)
            n_points_seen += 1
            n_curves = len(curves)
            if self.verbose:
                print(f'Points: {n_points_seen}; Curves: {n_curves}', end='\r')
            distances = [curve.get_distance(t, x) for curve in curves]
            if len(distances) == 0:
                curves.append(Curve(self.epsilon, alpha_prior, t, x))
                continue
            min_distance = np.min(distances)
            min_dist_cond = min_distance > self.distance_threshold
            time_cond = t < self.t_threshold
            if time_cond & min_dist_cond:
                curves.append(Curve(self.epsilon, alpha_prior, t, x))
            else:
                curve_index = np.argmin(distances)
                curves[curve_index].add_point(t, x)
        if self.verbose:
            print('Retrieving who belongs to who')
        curve_indexes = []
        for t in times:
            for curve_index, curve in enumerate(curves):
                if t in curve.times:
                    curve_indexes.append(curve_index)
        return curve_indexes
