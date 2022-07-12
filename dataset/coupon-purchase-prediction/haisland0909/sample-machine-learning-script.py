import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model.logistic import LogisticRegression

script_path = os.path.abspath(os.path.dirname(__file__))


class Get_Price_Rate(BaseEstimator, TransformerMixin):
    '''
    get price rate
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    def fit(self, date_frame, y=None):
        '''
        fit

        :param pandas.DataFrame: all data
        :rtype: Get_Price_Rate
        '''

        return self

    def transform(self, date_frame):
        '''
        transform

        :param pandas.DataFrame: all data
        :rtype: array
        '''

        return date_frame["PRICE_RATE"].as_matrix()[None].T.astype(np.float)


class Get_Match_Pref(BaseEstimator, TransformerMixin):
    '''
    get user pref is match coupon area
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    def fit(self, date_frame, y=None):
        '''
        fit

        :param pandas.DataFrame: all data
        :rtype: Get_Price_Rate
        '''

        return self

    def transform(self, date_frame):
        '''
        transform

        :param pandas.DataFrame: all data
        :rtype: array
        '''
        res_sr = date_frame["PREF_NAME"] == date_frame["ken_name"]

        return res_sr.as_matrix()[None].T.astype(np.float)


def top_merge(df, n=10, column="predict", merge_column="COUPON_ID_hash"):
    '''
    get top n row

    :param pandas.DataFrame df:
    :param int n:
    :param str column:
    :rtype: pandas.DataFrame
    '''

    return " ".join(df.sort_index(by=column)[-n:][merge_column])

feature_list = [
    ('PRICE_RATE', Get_Price_Rate()),
    ('MATCH_PREF', Get_Match_Pref()),
]

if __name__ == '__main__':
    # import csv
    user_df = pd.read_csv("%s/../input/user_list.csv" % script_path)
    train_coupon_df = pd.read_csv("%s/../input/coupon_list_train.csv" %
                                  script_path)
    train_visit_df = pd.read_csv("%s/../input/coupon_visit_train.csv" %
                                 script_path)
    test_coupon_df = pd.read_csv("%s/../input/coupon_list_test.csv" %
                                 script_path)
    # create train_df
    train_df = pd.merge(train_visit_df, train_coupon_df,
                        left_on="VIEW_COUPON_ID_hash", right_on="COUPON_ID_hash")
    train_df = pd.merge(train_df, user_df,
                        left_on="USER_ID_hash", right_on="USER_ID_hash")
    # create train feature
    fu_obj = FeatureUnion(transformer_list=feature_list)
    X_train = fu_obj.fit_transform(train_df)
    y_train = train_df["PURCHASE_FLG"]
    assert X_train.shape[0] == y_train.size
    # fit model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    # create test_df
    test_coupon_df["cross"] = 1
    user_df["cross"] = 1
    test_df = pd.merge(test_coupon_df, user_df, on="cross")
    # create test Feature
    X_test = fu_obj.transform(test_df)
    # predict test data
    predict_proba = clf.predict_proba(X_test)
    pos_idx = np.where(clf.classes_ == True)[0][0]
    test_df["predict"] = predict_proba[:, pos_idx]
    top10_coupon = test_df.groupby("USER_ID_hash").apply(top_merge)
    top10_coupon.name = "PURCHASED_COUPONS"
    top10_coupon.to_csv("submission.csv", header=True)
