from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 4
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.5
    num_rounds = 1000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

class CategoricalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column_name, k):
        self.threshold = k
        self.column_name = column_name

    def _reset(self):

        if hasattr(self, 'mapping_'):
            self.mapping_ = {}
            self.glob_med = 0
            self.glob_high = 0
            self.glob_low = 0

    def fit(self, X):

        self._reset()


        tmp = X.groupby([self.column_name, 'interest_level']).size().unstack().reset_index()

        tmp = tmp.fillna(0)
        tmp['record_count'] = tmp['high'] + tmp['medium'] + tmp['low']
        tmp['high_share'] = tmp['high']/tmp['record_count']
        tmp['med_share'] = tmp['medium']/tmp['record_count']

        self.glob_high = tmp['high'].sum()/tmp['record_count'].sum()
        self.glob_med = tmp['medium'].sum()/tmp['record_count'].sum()


        tmp['lambda'] = tmp['record_count'].apply(lambda x: 1.0 / (1.0 + math.exp(self.threshold-x)))

        tmp['w_high_' + self.column_name] = tmp[['high_share','lambda']].apply(lambda row: row['lambda']*row['high_share'] + (1.0-row['lambda'])*self.glob_high, axis=1)
        tmp['w_med_' + self.column_name] = tmp[['med_share','lambda']].apply(lambda row: row['lambda']*row['med_share'] + (1.0-row['lambda'])*self.glob_med, axis=1)

        tmp['w_high_' + self.column_name] = tmp['w_high_' + self.column_name].apply(lambda x : x*(1.0 + 0.01*(np.random.uniform()-0.5)))
        tmp['w_med_' + self.column_name] = tmp['w_med_' + self.column_name].apply(lambda x : x*(1.0 + 0.01*(np.random.uniform()-0.5)))


        self.mapping_ = tmp[['w_high_' + self.column_name, 'w_med_' + self.column_name,  self.column_name]]

        return self

    def transform(self, X):

        X = X.merge(self.mapping_.reset_index(), how = 'left', left_on = self.column_name, right_on = self.column_name)
        del X['index'] #remove the side-effect-of-merge column
        X['w_high_' + self.column_name] = X['w_high_' + self.column_name].apply(lambda x: x if not np.isnan(x) else self.glob_high)
        X['w_med_' + self.column_name] = X['w_med_' + self.column_name].apply(lambda x: x if not np.isnan(x) else self.glob_med)
        return X


def perform_general_feature_engineering(tr_df, te_df):

    for df in [tr_df, te_df]:

        df["room_sum"] = df["bedrooms"] + df["bathrooms"]

        df["price_bed"] = df["price"] / df["bedrooms"]
        df["price_t1"] = df["price"] / df["room_sum"]

        df["fold_t1"] = df["bedrooms"] / df["room_sum"]
        df['bath_room'] = df["bathrooms"]/df["bedrooms"]


        df["room_dif"] = df["bedrooms"] - df["bathrooms"]


        # count of photos #
        df["num_photos"] = df["photos"].apply(len)

        # count of "features" #
        df["num_features"] = df["features"].apply(len)

        # count of words present in description column #
        df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

        # convert the created column to datetime object so as to extract more features
        df["created"] = pd.to_datetime(df["created"])

        # Let us extract some features like year, month, day, hour from date columns #
        df["created_year"] = df["created"].dt.year
        df["created_month"] = df["created"].dt.month
        df['Zero_building_id'] = df['building_id'].apply(lambda x: 1 if x == '0' else 0)
        df['log_price'] = np.log(df['price'])

        df['num_exc'] = df['description'].apply(lambda x: len(x.split('!')))



    return tr_df, te_df

data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
train_df, test_df = perform_general_feature_engineering(train_df, test_df)
features_to_use = ['room_sum', 'price_bed', 'price_t1', 'fold_t1', 'bath_room', 'room_dif', 'num_photos', 'num_features', 'num_description_words', 'created_year', 'created_month', 'Zero_building_id', 'log_price', 'num_exc']

categorical = ["display_address", "building_id",'manager_id', "street_address", 'listing_id']
for f in categorical:
    if train_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values) + list(test_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))
        features_to_use.append(f)

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))



tfidf = CountVectorizer(stop_words='english', max_features=300)
tr_sparse = pd.DataFrame(tfidf.fit_transform(train_df["features"]).toarray(), index=train_df.index)
te_sparse = pd.DataFrame(tfidf.transform(test_df["features"]).toarray(), index= test_df.index)



#to simplify the feature encoding transformer fit method, interest_level passed as well
features_to_use.append('interest_level')
train_X = pd.concat([train_df[features_to_use], tr_sparse], axis=1)
features_to_use.remove('interest_level')
test_X = pd.concat([test_df[features_to_use], te_sparse], axis=1)

#get manager_id - interest_level distribution
man_encoder = CategoricalTransformer(column_name='manager_id', k=8)
train_X = man_encoder.fit_transform(train_X)
test_X = man_encoder.transform(test_X)

#get building_id - interest_level distribution
build_encoder = CategoricalTransformer(column_name='building_id', k=8)
train_X = build_encoder.fit_transform(train_X)
test_X = build_encoder.transform(test_X)

del train_X['interest_level']


target_num_map = {'high': 0, 'medium': 1, 'low': 2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))



preds, model = runXGB(train_X, train_y, test_X)

out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgb_submission.csv", index=False)