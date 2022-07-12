#! /usr/bin/env python3.4

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer


def concat_top_n(df, n=10):
    return ' '.join(df[:n]['COUPON_ID_hash'])


if __name__ == '__main__':
    userListDf = pd.read_csv('../input/user_list.csv')
    userListDf = userListDf[['USER_ID_hash', 'PREF_NAME']].fillna('missing')
    couponListTrainDf = pd.read_csv('../input/coupon_list_train.csv')
    couponListTestDf = pd.read_csv('../input/coupon_list_test.csv')
    allCouponListDf = couponListTrainDf.append(couponListTestDf)
    couponAreaTrainDf = pd.read_csv('../input/coupon_area_train.csv')
    couponAreaTestDf = pd.read_csv('../input/coupon_area_test.csv')
    allCouponAreaDf = couponAreaTrainDf.append(couponAreaTestDf)[['PREF_NAME']]
    couponVisitTrainDf = pd.read_csv('../input/coupon_visit_train.csv')
    couponVisitTrainDf = couponVisitTrainDf[['PURCHASE_FLG', 'VIEW_COUPON_ID_hash',
                                             'USER_ID_hash']].drop_duplicates()
    couponVisitTrainDf = couponVisitTrainDf.groupby(['VIEW_COUPON_ID_hash',
                                                     'USER_ID_hash']).sum().reset_index()
    trainDf = pd.merge(couponVisitTrainDf, allCouponListDf, left_on='VIEW_COUPON_ID_hash',
                       right_on='COUPON_ID_hash')
    trainDf = pd.merge(trainDf, userListDf, on='USER_ID_hash')

    y = trainDf['PURCHASE_FLG'].values

    features = trainDf[[
        'CAPSULE_TEXT',
        'CATALOG_PRICE',
        'ken_name',
        'PREF_NAME',
        'small_area_name',
    ]].values
    X = [
         {
             'capsule': capsule,
             'price': price,
             'ken': ken,
             'userKen': userKen,
             'smallArea': smallArea
         }
         for
         capsule,
         price,
         ken,
         userKen,
         smallArea
         in features]
    encoder = DictVectorizer()
    X = encoder.fit_transform(X)
    clf = DecisionTreeClassifier(class_weight='auto')
    clf = clf.fit(X, y)
    print('Featured:\t' + str(clf.score(X, y)))

    couponListTestDf['all'] = True
    userListDf['all'] = True
    testDf = pd.merge(userListDf, couponListTestDf, on='all')
    features = testDf[[
        'CAPSULE_TEXT',
        'CATALOG_PRICE',
        'ken_name',
        'PREF_NAME',
        'small_area_name',
    ]].values
    X = [
         {
             'capsule': capsule,
             'price': price,
             'ken': ken,
             'userKen': userKen,
             'smallArea': smallArea
         }
         for
         capsule,
         price,
         ken,
         userKen,
         smallArea
         in features]
    X = encoder.transform(X)
    probList = clf.predict_proba(X)
    labelTrueAt = np.where(clf.classes_ == 1)[0][0]
    testDf['PURCHASE_PROB'] = probList[:, labelTrueAt]
    testDf = testDf[['PURCHASE_PROB', 'DISCOUNT_PRICE', 'CATALOG_PRICE', 'USER_ID_hash',
                     'COUPON_ID_hash']]
    testDf = testDf.sort_index(by=['PURCHASE_PROB', 'DISCOUNT_PRICE', 'CATALOG_PRICE'],
                               ascending=[False, False, True])
    testDf = testDf[['USER_ID_hash', 'COUPON_ID_hash']]
    topN = testDf.groupby(['USER_ID_hash']).apply(concat_top_n)
    topN.name = 'PURCHASED_COUPONS'
    topN.to_csv("submission_tree.csv", header=True)
