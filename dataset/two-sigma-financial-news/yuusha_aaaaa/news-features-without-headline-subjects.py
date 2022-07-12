import gc
import itertools
import logging
import os
import re
import sys
from abc import abstractmethod, ABCMeta, ABC
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import  CountVectorizer
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

CATEGORY_START_END_PATTERN = r"[\{\}\']"

SPLIT_PATTERN = r"[{}']"

logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.addHandler(logging.FileHandler("main.log"))

try:
    TEST_MARKET_DATA = Path(__file__).parent.joinpath("data/test/marketdata_sample.csv")
    TEST_NEWS_DATA = Path(__file__).parent.joinpath("data/test/news_sample.csv")
except NameError as e:
    TEST_MARKET_DATA = "data/test/marketdata_sample.csv"
    TEST_NEWS_DATA = "data/test/news_sample.csv"

# MODEL_TYPE = "mlp"
MODEL_TYPE = "lgb"


class FeatureSetting(object):
    remove_news_features = ["headline", "subjects"]


def main():
    logger.info("This model type is {}".format(MODEL_TYPE))
    # coding: utf-8

    # leave only latest news
    # flat subjects
    # flat audiences
    # binary bow headline

    # ![](http://)![](http://)************# Two Sigma Financial News Competition Official Getting Started Kernel
    # ## Introduction
    #
    # This competition is different from most Kaggle Competitions in that:
    # * You can only submit from Kaggle Kernels, and you may not use other data sources, GPU, or internet access.
    # * This is a **two-stage competition**.  In Stage One you can edit your Kernels and improve your model, where Public Leaderboard scores are based on their predictions relative to past market data.  At the beginning of Stage Two, your Kernels are locked, and we will re-run your Kernels over the next six months, scoring them based on their predictions relative to live data as those six months unfold.
    # * You must use our custom **`kaggle.competitions.twosigmanews`** Python module.  The purpose of this module is to control the flow of information to ensure that you are not using future data to make predictions for the current trading day.
    #
    # ## In this Starter Kernel, we'll show how to use the **`twosigmanews`** module to get the training data, get test features and make predictions, and write the submission file.
    # ## TL;DR: End-to-End Usage Example
    # ```
    # from kaggle.competitions import twosigmanews
    # env = twosigmanews.make_env()
    #
    # (market_train_df, news_train_df) = env.get_training_data()
    # train_my_model(market_train_df, news_train_df)
    #
    # for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
    #   predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
    #   env.predict(predictions_df)
    #
    # env.write_submission_file()
    # ```
    # Note that `train_my_model` and `make_my_predictions` are functions you need to write for the above example to work.

    # In[1]:

    # In[ ]:

    # You can only call make_env() once, so don't lose it!
    env, market_train_df, news_train_df = load_train_dfs()

    # ## **`get_training_data`** function
    #
    # Returns the training data DataFrames as a tuple of:
    # * `market_train_df`: DataFrame with market training data
    # * `news_train_df`: DataFrame with news training data
    #
    # These DataFrames contain all market and news data from February 2007 to December 2016.  See the [competition's Data tab](https://www.kaggle.com/c/two-sigma-financial-news/data) for more information on what columns are included in each DataFrame.

    # In[ ]:

    compress_dtypes(market_train_df)

    # In[ ]:

    compress_dtypes(news_train_df)

    # In[ ]:

    news_train_df.tail()

    # # add necessary info

    # In[ ]:

    MARKET_ID = "id"
    NEWS_ID = "news_id"

    # In[ ]:

    add_ids(MARKET_ID, NEWS_ID, market_train_df, news_train_df)

    # In[ ]:

    market_train_df["id"].max()

    # In[ ]:

    news_train_df["news_id"].max()

    add_confidence(market_train_df)

    market_train_df.id.tail()

    # # full fill missing values

    # In[ ]:

    market_train_df.isnull().sum(axis=0)
    # In[ ]:

    news_train_df.isnull().sum(axis=0)

    # In[ ]:

    # empty string check
    for col, dtype in zip(news_train_df.columns, news_train_df.dtypes):
        if dtype == np.dtype('O'):
            n_empty = (news_train_df[col] == "").sum()
            logger.info("empty value in {}: {}".format(col, n_empty))

    # In[ ]:
    # In[ ]:

    # In[ ]:

    fill_missing_value_news_df(news_df=news_train_df)

    # In[ ]:

    # In[ ]:
    market_preprocess = MarketPreprocess()
    market_preprocess.fit_transform(market_train_df)
    news_preprocess = NewsPreprocess()
    news_preprocess.fit_transform(news_train_df)

    to_category_news_df(news_train_df)

    features = Features()
    features.fit(market_train_df, news_train_df)

    MAX_DAY_DIFF = 3

    gc.collect()

    # In[ ]:

    linker = MarketNewsLinker(Features.CONCATABLE_FEATURES, MAX_DAY_DIFF)
    linker.link(market_train_df, news_train_df)
    del news_train_df
    market_train_df = linker.create_new_market_df()
    linker.clear()
    gc.collect()

    # In[ ]:

    from collections import OrderedDict

    # # feature extraction II and dimension reduction

    news_features = OrderedDict()

    # In[ ]:

    news_feature_names, pipeline = Features.post_merge_feature_extraction(features, market_train_df)

    # In[ ]:

    gc.collect()

    # In[ ]:

    model = ModelWrapper.generate(MODEL_TYPE)

    # In[ ]:

    market_train_df, features.news = model.create_dataset(market_train_df, features.news, news_feature_names,
                                                          train_batch_size=400, valid_batch_size=400)
    model.train()
    from pandas.api.types import CategoricalDtype

    def to_category_type(df, category_columns, categories_list):
        for col, categories in zip(category_columns, categories_list):
            cat_type = CategoricalDtype(categories=categories)
            df[col] = df[col].astype(cat_type)

    # In[ ]:

    days = env.get_prediction_days()

    # In[ ]:

    # ## `get_prediction_days` function
    #
    # Generator which loops through each "prediction day" (trading day) and provides all market and news observations which occurred since the last data you've received.  Once you call **`predict`** to make your future predictions, you can continue on to the next prediction day.
    #
    # Yields:
    # * While there are more prediction day(s) and `predict` was called successfully since the last yield, yields a tuple of:
    #     * `market_observations_df`: DataFrame with market observations for the next prediction day.
    #     * `news_observations_df`: DataFrame with news observations for the next prediction day.
    #     * `predictions_template_df`: DataFrame with `assetCode` and `confidenceValue` columns, prefilled with `confidenceValue = 0`, to be filled in and passed back to the `predict` function.
    # * If `predict` has not been called since the last yield, yields `None`.

    # ### **`predict`** function
    # Stores your predictions for the current prediction day.  Expects the same format as you saw in `predictions_template_df` returned from `get_prediction_days`.
    #
    # Args:
    # * `predictions_df`: DataFrame which must have the following columns:
    #     * `assetCode`: The market asset.
    #     * `confidenceValue`: Your confidence whether the asset will increase or decrease in 10 trading days.  All values must be in the range `[-1.0, 1.0]`.
    #
    # The `predictions_df` you send **must** contain the exact set of rows which were given to you in the `predictions_template_df` returned from `get_prediction_days`.  The `predict` function does not validate this, but if you are missing any `assetCode`s or add any extraneous `assetCode`s, then your submission will fail.

    # Let's make random predictions for the first day:

    # In[ ]:

    predict_all(MARKET_ID, NEWS_ID, days, env, linker, market_preprocess, model, news_feature_names, news_preprocess,
                pipeline)

    # In[ ]:

    # In[ ]:

    logger.info('Done!')

    # In[ ]:

    env.write_submission_file()

    # In[ ]:

    # ## Main Loop
    # Let's loop through all the days and make our random predictions.  The `days` generator (returned from `get_prediction_days`) will simply stop returning values once you've reached the end.

    # @measure_time
    logger.info([filename for filename in os.listdir('.') if '.csv' in filename])

    #  ## **`write_submission_file`** function
    #
    # Writes your predictions to a CSV file (`submission.csv`) in the current working directory.

    # In[ ]:
    # We've got a submission file!

    # As indicated by the helper message, calling `write_submission_file` on its own does **not** make a submission to the competition.  It merely tells the module to write the `submission.csv` file as part of the Kernel's output.  To make a submission to the competition, you'll have to **Commit** your Kernel and find the generated `submission.csv` file in that Kernel Version's Output tab (note this is _outside_ of the Kernel Editor), then click "Submit to Competition".  When we re-run your Kernel during Stage Two, we will run the Kernel Version (generated when you hit "Commit") linked to your chosen Submission.

    # ## Restart the Kernel to run your code again
    # In order to combat cheating, you are only allowed to call `make_env` or iterate through `get_prediction_days` once per Kernel run.  However, while you're iterating on your model it's reasonable to try something out, change the model a bit, and try it again.  Unfortunately, if you try to simply re-run the code, or even refresh the browser page, you'll still be running on the same Kernel execution session you had been running before, and the `twosigmanews` module will still throw errors.  To get around this, you need to explicitly restart your Kernel execution session, which you can do by pressing the Restart button in the Kernel Editor's bottom Console tab:
    # ![Restart button](https://i.imgur.com/hudu8jF.png)

def measure_time(func):
    def inner(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        duration = perf_counter() - start
        logger.info("%s took %.6f sec", func.__name__, duration)
        return result

    return inner

class UnionFeaturePipeline(object):

    def __init__(self, *args):
        if args is None:
            self.transformers = []
        else:
            self.transformers = list(args)

    def transform(self, df, include_sparse=True):
        feature_columns = [transformer.transform(df) for transformer in self.transformers]
        feature_columns = [column for column in feature_columns if column is not None]
        if include_sparse:
            return sparse.hstack(feature_columns)
        return np.hstack(feature_columns)

    def add(self, transformer):
        self.transformers.append(transformer)


# In[ ]:
# TODO change make_column_transformer
def predict_all(MARKET_ID, NEWS_ID, days, env, linker, market_preprocess, model, news_feature_names, news_preprocess,
                pipeline):
    for (market_obs_df, news_obs_df, predictions_template_df) in tqdm(days):
        make_predictions(MARKET_ID, NEWS_ID, linker, market_preprocess, news_preprocess, pipeline,
                         news_feature_names, model, market_obs_df, news_obs_df,
                         predictions_template_df)
        env.predict(predictions_template_df)

def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0


def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0

class ModelWrapper(ABC):

    def __init__(self, **kwargs):
        self.model = None
        super().__init__(**kwargs)

    @abstractmethod
    def predict(self, X: np.ndarray):
        return None

    @abstractmethod
    def train(self, **kwargs):
        return self

    @staticmethod
    def generate(model_type):
        if model_type == "lgb":
            return LgbWrapper()
        elif model_type == "mlp":
            return MLPWrapper()
        else:
            raise ValueError("unknown model type: {}".format(model_type))

    @staticmethod
    def split_train_validation(market_obs_ids, train_X, train_Y, train_size, train_X2=None):
        train_size = int(train_X.shape[0] * train_size)
        orders = np.argsort(market_obs_ids).tolist()
        train_X = train_X[orders]
        train_Y = train_Y[orders]
        valid_X, valid_Y = train_X[train_size:], train_Y[train_size:]
        train_X, train_Y = train_X[:train_size], train_Y[:train_size]

        if train_X2 is not None:
            train_X2 = train_X2[orders]
            valid_X2 = train_X2[train_size:]
            train_X2 = train_X2[:train_size]
        else:
            valid_X2 = None

        return train_X, valid_X, train_Y, valid_Y, train_X2, valid_X2

    @staticmethod
    def to_x_y(df, news_feature_names):
        def to_Y(df):
            return np.asarray(df.confidence)

        train_Y = to_Y(df=df)
        df.drop(["confidence"], axis=1, inplace=True)
        market_obs_ids = df.id
        #     news_obs_ids = df.news_id
        #    market_obs_times = df.time
        df.drop(["id"], axis=1, inplace=True)
        feature_names = df.columns.tolist()
        feature_names.extend(news_feature_names)
        train_X = df.values
        del df
        gc.collect()
        return feature_names, market_obs_ids, train_X, train_Y

    @abstractmethod
    def create_dataset(self, market_train, features, news_feature_names, train_batch_size, valid_batch_size):
        return None, None


class LgbWrapper(ModelWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @measure_time
    def train(self, **kwargs):
        len(self.feature_names)
        gc.collect()
        # In[ ]:
        RANDOM_SEED = 10
        # In[ ]:

        hyper_params = {"objective": "binary", "boosting": "gbdt", "num_iterations": 500,
                        "learning_rate": 0.2, "num_leaves": 2500,
                        "num_threads": 2, "max_bin": 205, 'min_data_in_leaf': 210,
                        "seed": RANDOM_SEED, "early_stopping_round": 10
                        }
        # ## train
        # In[ ]:
        model = lgb.train(params=hyper_params, train_set=self.x, valid_sets=[self.valid_X])
        # In[ ]:
        for feature, imp in zip(model.feature_name(), model.feature_importance()):
            logger.info("{}: {}".format(feature, imp))
        # In[ ]:
        #
        del self.x
        # In[ ]:
        del self.valid_X
        # In[ ]:
        gc.collect()
        # In[ ]:
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def create_dataset(self, df, train_X2, feature_names, train_batch_size=None, valid_batch_size=None):
        self.feature_names, self.market_obs_ids, self.x, y = ModelWrapper.to_x_y(df, feature_names)
        train_size = 0.8
        self.x, self.valid_X, y, valid_Y, train_X2, valid_X2 = ModelWrapper.split_train_validation(self.market_obs_ids,
                                                                                                   self.x, y,
                                                                                                   train_size,
                                                                                                   train_X2)

        self.x = lgb.Dataset(sparse.hstack([self.x, train_X2]), label=y, feature_name=self.feature_names,
                             free_raw_data=False)
        del train_X2
        del y
        gc.collect()

        self.valid_X = self.x.create_valid(sparse.hstack([self.valid_X, valid_X2]), label=valid_Y)
        del valid_X2
        del valid_Y
        return None, None


class MarketNewsLinker(object):

    def __init__(self, concatable_fields, max_day_diff):
        self.market_df = None
        self.news_df = None
        self.link_dfs = None
        self.market_columns = None
        self.max_day_diff = max_day_diff
        self.datatypes_before_aggregation = None
        self.concatable_features = concatable_fields
        self.news_columns = None

    def link_market_assetCode_and_news_assetCodes(self):
        assetCodes_in_markests = self.market_df.assetCode.unique()
        logger.info("assetCodes pattern in markets: {}".format(len(assetCodes_in_markests)))
        assetCodes_in_news = self.news_df.assetCodes.unique()
        assetCodes_in_news_size = len(assetCodes_in_news)
        logger.info("assetCodes pattern in news: {}".format(assetCodes_in_news_size))
        parse_multiple_codes = lambda codes: re.sub(SPLIT_PATTERN, "", str(codes)).split(", ")
        parsed_assetCodes_in_news = [parse_multiple_codes(str(codes)) for codes in assetCodes_in_news]
        # len(max(parsed_assetCodes_in_news, key=lambda x: len(x)))
        all_assetCode_type_in_news = list(set(itertools.chain.from_iterable(assetCodes_in_news)))
        # check linking
        links_assetCodes = [[[raw_codes, market_assetCode] for parsed_codes, raw_codes in
                             zip(parsed_assetCodes_in_news, assetCodes_in_news) if
                             str(market_assetCode) in parsed_codes] for market_assetCode in assetCodes_in_markests]
        links_assetCodes = list(itertools.chain.from_iterable(links_assetCodes))
        logger.info("links for assetCodes: {}".format(len(links_assetCodes)))
        links_assetCodes = pd.DataFrame(links_assetCodes, columns=["newsAssetCodes", "marketAssetCode"],
                                        dtype='category')
        self.market_df = self.market_df.merge(links_assetCodes, left_on="assetCode", right_on="marketAssetCode")
        # merge assetCodes links
        self.market_df.drop(["assetCode", "marketAssetCode"], axis=1, inplace=True)
        self.market_columns.remove("assetCode")

    def append_working_date_on_market(self):
        self.market_df["date"] = self.market_df.time.dt.date
        self.news_df["firstCreatedDate"] = self.news_df.firstCreated.dt.date
        self.news_df.firstCreatedDate = self.news_df.firstCreatedDate.astype(np.datetime64)

        working_dates = self.news_df.firstCreatedDate.unique()
        working_dates.sort()
        market_dates = self.market_df.date.unique().astype(np.datetime64)
        market_dates.sort()

        def find_prev_date(date):
            for diff_day in range(1, self.max_day_diff + 1):
                prev_date = date - np.timedelta64(diff_day, 'D')
                if len(np.searchsorted(working_dates, prev_date)) > 0:
                    return prev_date
            return None

        prev_news_days_for_market_day = np.apply_along_axis(arr=market_dates, func1d=find_prev_date, axis=0)

        date_df = pd.DataFrame(columns=["date", "prevDate"])
        date_df.date = market_dates

        date_df.prevDate = prev_news_days_for_market_day

        self.market_df.date = self.market_df.date.astype(np.datetime64)
        self.market_df = self.market_df.merge(date_df, left_on="date", right_on="date")

    def link_market_id_and_news_id(self):
        logger.info("linking ids...")
        self.news_columns = self.news_df.columns.tolist()
        # merge market and news
        link_df = self.market_df.merge(self.news_df, left_on=["newsAssetCodes", "date"],
                                       right_on=["assetCodes", "firstCreatedDate"], how='left')
        # remove news after market obs
        link_df = link_df[link_df["time"] > link_df["firstCreated"]]
        # self.link_df.sort_values(by=["time"],inplace=True)
        # self.link_df.drop_duplicates(subset=["id"], keep="last", inplace=True)

        prev_day_link_df = self.market_df.merge(
            self.news_df, left_on=["newsAssetCodes", "prevDate"],
            right_on=["assetCodes", "firstCreatedDate"])
        self.market_df = pd.concat([link_df, prev_day_link_df])
        # self.market_df_prev_day_news.sort_values(by=["firstCreated"], inplace=True)

    def aggregate_day_asset_news(self):
        logger.info("aggregating....")
        agg_func_map = {column: "mean" for column in self.market_df.columns.tolist()
                        if column not in self.market_columns
                        and column not in self.concatable_features}
        agg_func_map.update({col: "first"
                             for col in self.market_columns})
        agg_func_map.update({
            column: lambda x: ", ".join(x) if x.dtype == "object" else ", ".join([str(v) for v in x])
            for column in self.concatable_features
        })
        agg_func_map["headline"] = lambda x: " ".join(x)
        #         agg_func_map["news_id"] = lambda x: x
        logger.info(agg_func_map)
        logger.info(self.market_df.dtypes)
        gc.collect()
        self.market_df = self.market_df.groupby("id").agg(agg_func_map)
        self._update_inner_data()
        self.fit_datatype()

    def _update_inner_data(self):
        self.market_columns = self.market_df.columns.tolist()

    def fit_datatype(self):
        for col in self.market_df.columns:
            previous_dtype = self.datatypes_before_aggregation[col]
            if previous_dtype == np.dtype('float16') or previous_dtype == np.dtype(
                    'int16') or previous_dtype == np.dtype('int8') or previous_dtype == np.dtype("bool"):
                self.market_df[col] = self.market_df[col].astype("float16")
            elif previous_dtype == np.dtype("float64") or previous_dtype == np.dtype(
                    "float32") or previous_dtype == np.dtype('int32'):
                self.market_df[col] = self.market_df[col].astype("float32")

    @measure_time
    def link(self, market_df, news_df):
        self.market_df = market_df
        self.news_df = news_df
        self.link_dfs = None

        self.market_columns = self.market_df.columns.tolist()
        self.datatypes_before_aggregation = {col: t for col, t in zip(self.market_columns, self.market_df.dtypes)}
        self.datatypes_before_aggregation.update(
            {col: t for col, t in zip(self.news_df.columns, self.news_df.dtypes)}
        )
        self.link_market_assetCode_and_news_assetCodes()

        self.append_working_date_on_market()

        return self.link_market_id_and_news_id()

    @measure_time
    def create_new_market_df(self):
        logger.info("updating market df....")
        dropped_columns = ["time", "date", "prevDate", "newsAssetCodes",
                           "assetCodes",
                           "firstCreated", "firstCreatedDate", "news_id"]
        self.market_df.drop(dropped_columns, axis=1, inplace=True)

        new_market_columns = self.market_df.columns.tolist()

        self.full_fill_new_columns(self.market_columns, new_market_columns)

        self.market_columns = self.market_df.columns.tolist()
        self.aggregate_day_asset_news()

        logger.info("linking done")
        return self.market_df

    def full_fill_new_columns(self, old_columns, new_columns):
        for col in set(new_columns) - set(old_columns):
            if "int" in str(self.market_df[col].dtype) or "float" in str(self.market_df[col].dtype):
                self.market_df[col].fillna(0)
            elif self.market_df[col].dtype == np.dtype("object"):
                self.market_df[col].fillna("")
            elif self.market_df[col].dtype == np.dtype("bool"):
                self.market_df[col].fillna(False)

    def clear(self):
        del self.market_df
        self.market_df = None
        self.news_df = None
        self.link_dfs = None
        self.market_columns = None
        self.datatypes_before_aggregation = None


def make_predictions(MARKET_ID, NEWS_ID, linker, market_preprocess, news_preprocess, pipeline, news_feature_names,
                     model: ModelWrapper, market_obs_df, news_obs_df,
                     predictions_df):
    add_ids(MARKET_ID, NEWS_ID, market_obs_df, news_obs_df)
    fill_missing_value_news_df(news_obs_df)
    compress_dtypes(market_obs_df)
    compress_dtypes(news_obs_df)
    #     to_category_type(news_obs_df, category_columns=categorical_features,
    #                      categories_list= news_categories)
    #     encode_categorical_fields(news_df=news_obs_df)
    remove_unnecessary_columns(market_obs_df, news_obs_df)

    market_preprocess.transform(market_obs_df)
    news_preprocess.transform(news_obs_df)
    linker.link(market_obs_df, news_obs_df)
    del news_obs_df
    gc.collect()
    market_obs_df = linker.create_new_market_df()
    news_features = pipeline.transform(market_obs_df)
    X, market_train_obs_ids, news_train_obs_ids, market_train_obs_times, feature_names = to_X(market_obs_df,
                                                                                              None, [],
                                                                                              news_features,
                                                                                              news_feature_names)

    predictions = model.predict(X)
    predictions_df.confidenceValue[[market_id - 1 for market_id in market_train_obs_ids]] = predictions * 2 - 1


def to_X(df, news_features, news_feature_names, additional_feature, additional_feature_names):
    market_obs_ids = df.id
    #     news_obs_ids = df.news_id
    news_obs_ids = []
    #    market_obs_times = df.time
    market_obs_times = []
    df.drop(["id"], axis=1, inplace=True)

    feature_names = df.columns.tolist()

    logger.info(df.dtypes)

    if len(news_feature_names) > 0 and isinstance(news_features, np.ndarray):
        feature_names.extend(news_feature_names)
        row_indices = [market_id - 1 for market_id in market_obs_ids]
        news_features = news_features[row_indices]
        X = np.hstack([df.values, news_features])
        del news_features

    elif len(additional_feature_names) > 0 and (
            isinstance(additional_feature, np.ndarray) or sparse.issparse(additional_feature)):
        feature_names.extend(additional_feature_names)
        if sparse.issparse(additional_feature):
            X = sparse.hstack([df.values, additional_feature]).tocsr()
        else:
            X = np.hstack([df.values, additional_feature])
    else:
        raise TypeError("Unknown type to concat")
    del df
    gc.collect()
    return X, market_obs_ids, news_obs_ids, market_obs_times, feature_names


def to_category_news_df(news_df):
    news_df.headlineTag = news_df.headlineTag.astype('category')


def fill_missing_value_news_df(news_df):
    news_df.headlineTag.replace("", "UNKNONWN", inplace=True)


def add_confidence(df):
    # TODO change confidence by return proportion
    df["confidence"] = df["returnsOpenNextMktres10"] >= 0


def add_ids(market_id, news_id, market_df, news_df):
    add_id(market_df, market_id)
    add_id(news_df, news_id)


def add_id(df, id_name):
    df[id_name] = df.index.astype("int32") + 1


def compress_dtypes(news_df):
    for col, dtype in zip(news_df.columns, news_df.dtypes):
        if dtype == np.dtype('float64'):
            news_df[col] = news_df[col].astype("float32")
        if dtype == np.dtype('int64'):
            news_df[col] = news_df[col].astype("int32")


def load_train_dfs():
    try:
        from kaggle.competitions import twosigmanews
        env = twosigmanews.make_env()
        (market_train_df, news_train_df) = env.get_training_data()
    except:
        market_train_df = pd.read_csv(TEST_MARKET_DATA, encoding="utf-8", engine="python")
        news_train_df = pd.read_csv(TEST_NEWS_DATA, encoding="utf-8", engine="python")
        env = None
    return env, market_train_df, news_train_df


def remove_unnecessary_columns(market_df, news_df):
    # market_df.drop(["returnsOpenNextMktres10", "universe"], axis=1, inplace=True)
    market_df.drop(["assetName"], axis=1, inplace=True)
    news_df.drop(['time', 'sourceId', 'sourceTimestamp', "assetName"], axis=1, inplace=True)


def remove_unnecessary_columns_train(market_df, news_df):
    market_df.drop(["returnsOpenNextMktres10", "universe"], axis=1, inplace=True)
    remove_unnecessary_columns(market_df, news_df)


class TorchDataset(Dataset):
    def __init__(self, matrix, labels, transformers=None):
        self._matrix = matrix
        self._labels = labels
        self._transformers = transformers
        self.n_features = matrix.shape[-1]

    def __getitem__(self, index):
        item = self._matrix[index, :]
        if self._transformers is None:
            return item
        return self._transformers(item), torch.Tensor(self._labels[index:index + 1])

    def __len__(self):
        return self._matrix.shape[0]


class TorchDataLoader(DataLoader):

    def __init__(self, dataset: TorchDataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0,
                 collate_fn=default_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn)

    def __len__(self):
        return len(self.dataset)


def create_data_loader(matrix: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    def transformers(item):
        return torch.from_numpy(item.todense())

    dataset = TorchDataset(matrix, labels, transformers)

    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class BaseMLPClassifier(nn.Module):

    def __init__(self, fc_layer_params: list):
        super().__init__()
        layers = [
            nn.Sequential(
                nn.Linear(**params),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
            for i, params in enumerate(fc_layer_params[:-1])
        ]
        # TODO add dropout
        self.fc_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(**fc_layer_params[-1])
        # if self.output_layer.out_features == 1:
        self.sigmoid = nn.Sigmoid()
        # # self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc_layers(x)
        out = self.output_layer(out)
        # if self.output_layer.out_features == 1:
        out = self.sigmoid(out)
        # out = self.softmax(out)
        return out


class BaseMLPTrainer(object):
    def __init__(self, model, loss_function, score_function, optimizer_factory):
        self.model: nn.Module = model
        # self.loss_function = nn.BCELoss()
        self.loss_function = loss_function
        self.score_function = score_function

        self.optimiser = optimizer_factory(self.model)

        self.train_data_loader = None
        self.valid_data_loader = None

        self.n_epoch = None
        self._current_epoch = 0
        self.train_losses = []
        self.train_scores = []
        self.valid_losses = []
        self.valid_scores = []

        self._current_max_valid_score = 0
        self._early_stop_count = 0

        self.save_name = "twosigma.model"

    def train(self, train_data_loader, valid_data_loader, n_epochs):
        self.clear_history()

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.n_epoch = n_epochs

        iterator = tqdm(range(n_epochs))
        for epoch in iterator:
            self._current_epoch = epoch + 1
            logger.info("training %d epoch / n_epochs", self._current_epoch)

            self._train_epoch()
            self._valid_epoch()

            if self.valid_scores[-1] <= self._current_max_valid_score:
                self._early_stop_count += 1
            else:
                logger.info("validation score is improved from %.3f to %.3f",
                            self._current_max_valid_score, self.valid_scores[-1])
                self._current_max_valid_score = self.valid_scores[-1]
                self._early_stop_count = 0
                self.save_models()

            if self._early_stop_count >= 10:
                logger.info("======early stopped=====")
                self.model.load_state_dict(torch.load(self.save_name))
                iterator.close()
                break

        logger.info("train done!")

    def clear_history(self):
        self.n_epoch = None
        self._current_epoch = 0

        self.train_losses = []
        self.train_scores = []
        self.valid_losses = []
        self.valid_scores = []

        self._current_max_valid_score = 0
        self._early_stop_count = 0

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for i, data in enumerate(self.valid_data_loader):
            inputs, labels = data

            self.optimiser.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)

            loss.backward()
            self.optimiser.step()
            total_loss += loss.item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] loss: %.3f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))

        avg_loss = total_loss / len(self.train_data_loader)
        logger.info("******train loss at epoch %d: %.3f :" % (self._current_epoch, avg_loss))
        self.train_losses.append(avg_loss)

    def _valid_epoch(self):
        total_loss = 0.0

        all_labels = []
        all_outputs = []
        self.model.eval()
        for i, data in enumerate(self.valid_data_loader):
            inputs, labels = data
            outputs = self.model(inputs)
            all_labels.append(labels)
            all_outputs.append(outputs)
            loss = self.loss_function(outputs, labels)

            total_loss += loss.item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] validation loss: %.3f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))

        avg_loss = total_loss / len(self.valid_data_loader)
        self.valid_losses.append(avg_loss)
        logger.info("******valid loss at epoch %d: %.3f :" % (self._current_epoch, avg_loss))

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        score = self.score_function(all_outputs, all_labels)
        logger.info("******valid score at epoch %d: %.3f :" % (self._current_epoch, score))
        self.valid_scores.append(score)

    def save_models(self):
        torch.save(self.model.state_dict(), self.save_name)
        logger.info("Checkpoint saved")


class MLPWrapper(ModelWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, x: Union[np.ndarray, sparse.spmatrix]):
        logger.info("predicting %d samples...".format(x.shape[0]))
        self.model.eval()
        if sparse.issparse(x):
            x = x.todense()
        x = torch.from_numpy(x)
        return self.model(x).detach().numpy()

    def train(self, **kwargs):
        classes = 1
        model = BaseMLPClassifier(
            [{"in_features": self.train_data_loader.dataset.n_features, "out_features": 256, "bias": True},
             {"in_features": 256, "out_features": 64, "bias": True},
             {"in_features": 64, "out_features": 16, "bias": True},
             {"in_features": 16, "out_features": classes, "bias": True},
             ]
        )

        def score_function(predicted: torch.Tensor, labels: torch.Tensor):
            return roc_auc_score(labels.detach().numpy(), predicted.detach().numpy().reshape((-1)))

        optimizer_factory = lambda model: optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
        trainer = BaseMLPTrainer(model, loss_function=nn.BCELoss(),
                                 score_function=score_function,
                                 optimizer_factory=optimizer_factory)

        trainer.train(self.train_data_loader, self.valid_data_loader, 50)

        self.model = model
        return self

    def create_dataset(self, market_train, features, news_feature_names, train_batch_size, valid_batch_size):
        feature_names, market_obs_ids, market_train, labels = ModelWrapper.to_x_y(market_train, news_feature_names)
        market_train = sparse.hstack([market_train, features], format="csr")
        market_train, valid_matrix, labels, valid_labels, _, _ = ModelWrapper.split_train_validation(market_obs_ids,
                                                                                                     market_train,
                                                                                                     labels,
                                                                                                     train_size=0.8)
        self.train_data_loader = create_data_loader(market_train, labels, batch_size=train_batch_size, shuffle=True)
        self.valid_data_loader = create_data_loader(valid_matrix, valid_labels, batch_size=valid_batch_size,
                                                    shuffle=True)
        return None, None


def flatten_category_complex(cat_values):
    flat_cats = [re.sub(CATEGORY_START_END_PATTERN, "", value).split(", ")
                 for value in cat_values]
    return flat_cats


# Data

def binary_encode_cat_complex(encoder, target):
    return encoder.transform(flatten_category_complex(target))


class Preprocess(object):

    def __init__(self):
        self.transformers = [
            (col, LogTransformer(), col) for col in self.get_log_normal_columns()
        ]
        # self.transformers.extend([(col, TahnEstimators(), col) for col in self.get_columns_scaled()])
        # self.transformers.extend([(col, ReshapeInto2d(), col) for col in self.get_columns_scaled()])
        self.transformers.extend([(col, MinMaxScaler(), col) for col in self.get_columns_scaled()])

    def get_columns_scaled(self):
        return []

    def get_log_normal_columns(self):
        return []

    # inplace transform
    def fit_transform(self, df: pd.DataFrame):
        for new_col_name, transformer, col_name in self.transformers:
            if not new_col_name:
                new_col_name = col_name
            df[new_col_name] = transformer.fit_transform(to_2d_array(df[col_name]))

    def transform(self, df: pd.DataFrame):
        for new_col_name, transformer, col_name in self.transformers:
            if not new_col_name:
                new_col_name = col_name
            df[new_col_name] = transformer.transform(to_2d_array(df[col_name]))


class MarketPreprocess(Preprocess):
    COLUMNS_NORMALIZED = ['volume', 'close', 'open',
                          'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                          'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                          'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                          'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    # LOG_NORMAL_FIELDS = ['volume', 'close', 'open',
    #                      'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
    #                      'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
    #                      'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
    #                      'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    LOG_NORMAL_FIELDS = []

    def get_columns_scaled(self):
        return self.COLUMNS_NORMALIZED

    def get_log_normal_columns(self):
        return self.LOG_NORMAL_FIELDS


class NewsPreprocess(Preprocess):
    COLUMNS_SCALED = [
        'urgency', 'takeSequence',
        'bodySize', 'companyCount',
        'sentenceCount', 'wordCount',
        'firstMentionSentence', 'relevance', 'sentimentClass',
        'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
        'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
        'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
        'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
        'volumeCounts7D']
    LOG_NORMAL_FIELDS = [
        'bodySize',
        'sentenceCount', 'wordCount',
        'firstMentionSentence',
        'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
        'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
        'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
        'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
        'volumeCounts7D']

    def get_columns_scaled(self):
        return self.COLUMNS_SCALED

    def get_log_normal_columns(self):
        return self.LOG_NORMAL_FIELDS

class ReshapeInto2d(FunctionTransformer):
    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(to_2d_array, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)


def to_2d_array(x):
    array = x
    if isinstance(x, pd.Series):
        array = array.values
    if len(array.shape) == 1:
        array = array.reshape((-1, 1))
    return array


class LogTransformer(FunctionTransformer):

    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(LogTransformer.to_log, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    @staticmethod
    def to_log(x):
        input_ = x
        # input_ = input_
        return np.log1p(input_)


class TahnEstimators(BaseEstimator, TransformerMixin):
    """
    refer
    https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
    https://stackoverflow.com/questions/43061120/tanh-estimator-normalization-in-python
    """

    def __init__(self):
        self.std_ = None
        self.mean_ = None
        self.n_seen_samples = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        return self

    def transform(self, X, copy=None):
        return 0.5 * (np.tanh(0.01 * (to_2d_array(X) - self.mean_) / self.std_) + 1)


class Features(object):
    @staticmethod
    def post_merge_feature_extraction(features, market_train_df):
        pipeline = UnionFeaturePipeline()
        # ## audience
        # In[ ]:
        audience_transformer = FlatAudience(features.audience_encoder)
        # In[ ]:
        flat_audience_feature = audience_transformer.transform(market_train_df)
        # In[ ]:
        features.news["audiences"] = flat_audience_feature
        # In[ ]:
        pipeline.add(audience_transformer)
        # In[ ]:
        logger.info("audiences feature extraction has done.")
        # In[ ]:
        # sys.getsizeof(pipeline)
        # # provider
        # In[ ]:
        provider_transformer = ProviderBinaryEncode(features.provider_encoder)
        # In[ ]:
        provider_binary = provider_transformer.transform(market_train_df)
        # In[ ]:
        features.news["provider"] = provider_binary
        # In[ ]:
        gc.collect()
        # In[ ]:
        pipeline.add(provider_transformer)
        # In[ ]:
        logger.info("provider feature extraction has done.")
        # # headlineTag
        # In[ ]:
        headlineTag_transformer = HeadlineTagBinaryEncode(features.headlineTag_encoder)
        # In[ ]:
        headlineTag_category = headlineTag_transformer.transform(market_train_df)
        # In[ ]:
        features.news["headlineTag"] = headlineTag_category
        # In[ ]:
        pipeline.add(headlineTag_transformer)
        # In[ ]:
        logger.info("headlineTag feature extraction has done.")
        # # headline
        dropped_columns = []
        if "headline" not in FeatureSetting.remove_news_features:
            # In[ ]:

            headline_vectorizer = CountVectorizer(decode_error="ignore",
                                                  stop_words="english",
                                                  strip_accents="unicode",
                                                  max_features=3000,
                                                  binary=True,
                                                  dtype='int8')

            # In[ ]:

            binary_bow = headline_vectorizer.fit_transform(market_train_df.headline)

            # In[ ]:
            # In[ ]:

            headline_vectorizer.get_feature_names()

            # In[ ]:

            binary_bow[:10].todense()

            # In[ ]:

            # In[ ]:

            logger.info("headline feature extraction has done.")

            # In[ ]:

            pipeline.add(HeadlineBinaryBow(headline_vectorizer))

            # In[ ]:

            features.news["headline"] = binary_bow
        else:
            dropped_columns.append("headline")

        market_train_df.drop(["headline"], axis=1, inplace=True)

        gc.collect()

        # In[ ]:
        if "subjects" not in FeatureSetting.remove_news_features:
            subjects_encoder = CountVectorizer(decode_error="ignore",
                                               strip_accents="unicode",
                                               max_features=5000,
                                               binary=True,
                                               dtype='int8')

            # ## subjects

            # In[ ]:

            flat_subjects_category = subjects_encoder.fit_transform(market_train_df.subjects)

            # In[ ]:

            # subjects_transformer = FlatSubjects()

            # In[ ]:

            features.news["subjects"] = flat_subjects_category

            # In[ ]:

            pipeline.add(SubjectsBinary(subjects_encoder))

            # In[ ]:

            # In[ ]:

            logger.info("subjects feature extraction has done")
        else:
            dropped_columns.append("subjects")

        if len(dropped_columns) > 0:
            pipeline.add(DropColumnsTransformer(dropped_columns))
        
        market_train_df.drop(["subjects"], axis=1, inplace=True)

        market_train_df.to_csv("feature_df.csv")
        # In[ ]:
        # In[ ]:
        news_feature_names = [["{}_{}".format(name, i) for i in range(feature.shape[1])] for name, feature
                              in features.news.items()]
        # # convert into trainable form
        # In[ ]:
        # In[ ]:
        # market_train_df.drop(["confidence"], axis=1, inplace=True)
        news_feature_names = list(itertools.chain.from_iterable(news_feature_names))
        # In[ ]:
        features.news = sparse.hstack(list(features.news.values()), dtype="int8").tocsr()
        return news_feature_names, pipeline
    CONCATABLE_FEATURES = ["subjects", "audiences", "headline", "provider", "headlineTag"]

    def __init__(self):
        self.headlineTag_encoder = None
        self.provider_encoder = None
        self.audience_encoder = None
        self.news = OrderedDict()

    def fit(self, market_train_df: pd.DataFrame, news_train_df: pd.DataFrame):
        self.audience_encoder = generate_flatten_one_hot_encoder(news_train_df.audiences)

        # In[ ]:
        # subjects_encoder = generate_flatten_one_hot_encoder(news_train_df.subjects.cat.categories.values)
        # In[ ]:
        # subjects_encoder.classes_
        # In[ ]:
        self.provider_encoder = MultiLabelBinarizer(sparse_output=True)

        # ## flatten audience
        # In[ ]:
        self.provider_encoder.fit([news_train_df.provider.cat.categories.values])

        # ## encode provider
        # In[ ]:
        # In[ ]:
        # In[ ]:
        # In[ ]:
        self.headlineTag_encoder = MultiLabelBinarizer(sparse_output=True)
        self.headlineTag_encoder.fit([news_train_df.headlineTag.cat.categories.values])

        # # TODO flatten sentimentClass

        remove_unnecessary_columns_train(market_train_df, news_train_df)
        gc.collect()
        log_object_sizes()

        # headline = news_train_df.headline
        # news_train_df.drop(["headline"], axis=1, inplace=True)
        # subjects = news_train_df.subjects
        # news_train_df.drop(["subjects"], axis=1, inplace=True)
        # audiences = news_train_df.audiences
        # news_train_df.drop(["audiences"], axis=1, inplace=True)
        # categorical_column_values = [news_train_df[feature] for feature in categorical_features]
        # news_train_df.drop(categorical_features, axis=1, inplace=True)

def log_object_sizes():
    for memory_info in ["{}: {}".format(v, sys.getsizeof(eval(v)) / 1000000000) for v in dir()]:
        logger.info(memory_info)


def parse_category_complex(cat_value):
    return re.sub(CATEGORY_START_END_PATTERN, "", cat_value).split(", ")


def generate_flatten_one_hot_encoder(cat_values):
    flat_cats = flatten_category_complex(cat_values)
    flat_cats = list(set(itertools.chain.from_iterable(flat_cats)))
    flat_cats = np.asarray(list(flat_cats)).reshape(-1, 1)
    logger.info("flat category size: {}".format(len(flat_cats)))
    encoder = MultiLabelBinarizer(sparse_output=True)
    encoder.fit(flat_cats)
    return encoder


class FeatureTransformer(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, df):
        pass

    @abstractmethod
    def release_raw_field(self, df):
        pass


class FlatAudience(FeatureTransformer):

    def __init__(self, encoder):
        super(FlatAudience, self).__init__()
        self.encoder = encoder

    def transform(self, df):
        seq = df.audiences.astype("str")
        self.release_raw_field(df)
        seq = seq.apply(parse_category_complex)
        return self.encoder.transform(seq.tolist())

    def release_raw_field(self, df):
        df.drop(["audiences"], axis=1, inplace=True)
        gc.collect()


class SubjectsBinary(FeatureTransformer):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def transform(self, df):
        seq = df.subjects
        self.release_raw_field(df)
        return self.encoder.transform(seq)

    def release_raw_field(self, df):
        df.drop(["subjects"], axis=1, inplace=True)
        gc.collect()


class ProviderBinaryEncode(FeatureTransformer):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def transform(self, df):
        seq = df.provider.astype("str")
        self.release_raw_field(df)
        seq = seq.apply(parse_category_complex)
        return self.encoder.transform(seq)

    def release_raw_field(self, df):
        df.drop(["provider"], axis=1, inplace=True)
        gc.collect()


class HeadlineTagBinaryEncode(FeatureTransformer):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def transform(self, df):
        seq = df.headlineTag.astype("str")
        self.release_raw_field(df)
        seq = seq.apply(parse_category_complex)
        return self.encoder.transform(seq)

    def release_raw_field(self, df):
        df.drop(["headlineTag"], axis=1, inplace=True)
        gc.collect()


class HeadlineBinaryBow(FeatureTransformer):

    def __init__(self, encoder):
        self.headline_vectorizer = encoder

    def transform(self, df):
        seq = df.headline
        self.release_raw_field(df)
        return self.headline_vectorizer.transform(seq)

    def release_raw_field(self, df):
        df.drop(["headline"], axis=1, inplace=True)
        gc.collect()


class NullTransformer(FeatureTransformer):
    def transform(self, df):
        pass

    def release_raw_field(self, df):
        pass


class DropColumnsTransformer(NullTransformer):

    def __init__(self, columns):
        self.columns = columns

    def transform(self, df):
        df.drop(self.columns, axis=1, inplace=True)
        gc.collect()



if __name__ == '__main__':
    logger = logging.getLogger("root")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler("main.log"))
    main()
