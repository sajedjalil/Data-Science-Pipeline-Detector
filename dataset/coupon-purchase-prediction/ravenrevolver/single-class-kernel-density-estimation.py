import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

class Get_Price_Rate(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, date_frame, y=None):
        return self

    def transform(self, date_frame):
        return date_frame["PRICE_RATE"].as_matrix()[None].T.astype(np.float)


class Get_Match_Pref(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, date_frame, y=None):
        return self

    def transform(self, date_frame):
        res_sr = date_frame["PREF_NAME"] == date_frame["ken_name"]
        return res_sr.as_matrix()[None].T.astype(np.float)

class Get_Days_Join(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, date_frame, y=None):
        return self

    def transform(self, date_frame):
        coupon_start = pd.to_datetime(date_frame["DISPFROM"])
        user_join = pd.to_datetime(date_frame["REG_DATE"])
        days_join = user_join - coupon_start
        return days_join.as_matrix()[None].T.astype(np.float)

class Get_Gender(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, date_frame, y=None):
        return self

    def transform(self, date_frame):
        gender = date_frame["SEX_ID"] == 'M'
        return gender.as_matrix()[None].T.astype(np.float)

def top_merge(df, n=10, column="predict", merge_column="COUPON_ID_hash"):
    return " ".join(df.sort_index(by=column)[-n:][merge_column])


feature_list = [
    ('PRICE_RATE', Get_Price_Rate()),
    ('MATCH_PREF', Get_Match_Pref()),
    ('DAYS_JOIN', Get_Days_Join()),
    ('GENDER', Get_Gender())
]

if __name__ == '__main__':
    # read csv file
    user_df = pd.read_csv("../input/user_list.csv")
    train_coupon_detail_df = pd.read_csv("../input/coupon_detail_train.csv")
    train_coupon_list_df = pd.read_csv("../input/coupon_list_train.csv")
    test_coupon_df = pd.read_csv("../input/coupon_list_test.csv")
    # training data
    train_df = pd.merge(train_coupon_list_df, train_coupon_detail_df, left_on="COUPON_ID_hash", right_on="COUPON_ID_hash")
    train_df = pd.merge(train_df, user_df, left_on="USER_ID_hash", right_on="USER_ID_hash")
    # feature extraction for training data
    fu_obj = FeatureUnion(transformer_list=feature_list)
    X_train = fu_obj.fit_transform(train_df)
    # fit model
    kde = KernelDensity()
    kde.fit(X_train)
    # testing data
    test_coupon_df["cross"] = 1
    user_df["cross"] = 1
    test_df = pd.merge(test_coupon_df, user_df, on="cross")
    # feature extraction for testing data
    X_test = fu_obj.transform(test_df)
    # prediction for testing data
    score = np.exp(kde.score_samples(X_test))
    test_df["predict"] = score
    top10_coupon = test_df.groupby("USER_ID_hash").apply(top_merge)
    top10_coupon.name = "PURCHASED_COUPONS"
    top10_coupon.to_csv("submission.csv", header=True)
