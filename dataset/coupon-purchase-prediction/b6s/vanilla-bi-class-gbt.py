#! /usr/bin/env python3.4

import pandas as pd
import numpy as np
# from sklearn.cross_validation import cross_val_score
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


def concat_top_n(df, n=5):
    return ' '.join(df[:n]['COUPON_ID_hash'])


if __name__ == '__main__':
    couponVisitTrainDf = pd.read_csv('../input/coupon_visit_train.csv')[[
                            'PURCHASE_FLG', 'VIEW_COUPON_ID_hash',
                            'USER_ID_hash', 'PURCHASEID_hash']].drop_duplicates()
    couponDetailTrainDf = pd.read_csv('../input/coupon_detail_train.csv')
    couponTrainDf = pd.merge(couponVisitTrainDf, couponDetailTrainDf,
                             how='left',
                             on=['PURCHASEID_hash', 'USER_ID_hash']
                             ).fillna(0)[['USER_ID_hash', 'VIEW_COUPON_ID_hash',
                                          'PURCHASE_FLG', 'ITEM_COUNT',
                                          'PURCHASEID_hash']]
    couponTrainDf = couponTrainDf.groupby(['USER_ID_hash',
                                           'VIEW_COUPON_ID_hash']
                                          ).sum().reset_index()
    couponListTrainDf = pd.read_csv('../input/coupon_list_train.csv')
    couponTrainDf = pd.merge(couponListTrainDf, couponTrainDf,
                             left_on='COUPON_ID_hash',
                             right_on='VIEW_COUPON_ID_hash').fillna(-1)
    userListDf = pd.read_csv('../input/user_list.csv')
    userListDf = userListDf.loc[userListDf['WITHDRAW_DATE'].isnull()]
    trainDf = pd.merge(couponTrainDf, userListDf, on='USER_ID_hash').fillna('N/A')

    y = trainDf['PURCHASE_FLG'].values

    couponListTestDf = pd.read_csv('../input/coupon_list_test.csv')

    print('data loaded.')

    encoder = DictVectorizer(sparse=False, sort=False)

    features = trainDf[[
        'DISPPERIOD',
        'VALIDPERIOD',
        # 'USABLE_DATE_MON',
        # 'USABLE_DATE_TUE',
        # 'USABLE_DATE_WED',
        # 'USABLE_DATE_THU',
        # 'USABLE_DATE_FRI',
        # 'USABLE_DATE_SAT',
        # 'USABLE_DATE_SUN',
        'USABLE_DATE_HOLIDAY',
        'USABLE_DATE_BEFORE_HOLIDAY',
        'PRICE_RATE',
        'CATALOG_PRICE',
        'DISCOUNT_PRICE',
        'PREF_NAME',
        'ken_name',
        'small_area_name',
        'GENRE_NAME',
        'CAPSULE_TEXT',
        'AGE',
        'SEX_ID'
    ]].values
    X = [{
             'show': listPeriod if listPeriod < 60 else listPeriod / 60,
             'valid': validPeriod if validPeriod < 60 else validPeriod / 60,
            #  'sun': usableOnSunday,
             'ho': usableOnHoliday,
             'be': usableBeforeHoliday,
             'rate': rate,
             'price': price,
             'discount': discount,
             'ken': ken,
             'userKen': userKen,
             'ward': smallArea,
             'genre': genre,
             'type': capsule,
             'age': age,
             'sex': 1 if 'm' == sex else 0,
         }
         for
         listPeriod,
         validPeriod,
        #  usableOnMonday,
        #  usableOnTuesday,
        #  usableOnWednesday,
        #  usableOnThursday,
        #  usableOnFriday,
        #  usableOnSaturday,
        #  usableOnSunday,
         usableOnHoliday,
         usableBeforeHoliday,
         rate,
         price,
         discount,
         userKen,
         ken,
         smallArea,
         genre,
         capsule,
         age,
         sex
         in features]
    X = encoder.fit_transform(X)
    print('vectorized.')

    clf = RandomForestClassifier(min_samples_split=1, n_jobs=-1, verbose=2)
    # scores = cross_val_score(clf, X, y, n_jobs=-1, verbose=2)
    # print('Score:\t' + str(scores.mean()))
    clf = clf.fit(X, y)
    print('Score:\t' + str(clf.score(X, y)))

    # couponListTestDf['all'] = True
    # userListDf['all'] = True
    # testDf = pd.merge(userListDf, couponListTestDf, on='all')
    # features = testDf[[
    #     # 'DISPPERIOD',
    #     # 'VALIDPERIOD',
    #     # 'USABLE_DATE_MON',
    #     # 'USABLE_DATE_TUE',
    #     # 'USABLE_DATE_WED',
    #     # 'USABLE_DATE_THU',
    #     # 'USABLE_DATE_FRI',
    #     # 'USABLE_DATE_SAT',
    #     # 'USABLE_DATE_SUN',
    #     # 'USABLE_DATE_HOLIDAY',
    #     # 'USABLE_DATE_BEFORE_HOLIDAY',
    #     'CATALOG_PRICE',
    #     'DISCOUNT_PRICE',
    #     'PREF_NAME',
    #     'ken_name',
    #     # 'small_area_name',
    #     'GENRE_NAME',
    #     # 'CAPSULE_TEXT',
    #     # 'AGE',
    #     'SEX_ID'
    # ]].values
    # X = [{
    #         #  'show': listPeriod,
    #         #  'valid': validPeriod,
    #         #  'mon': usableOnMonday,
    #         #  'tue': usableOnTuesday,
    #         #  'wed': usableOnWednesday,
    #         #  'thu': usableOnThursday,
    #         #  'fri': usableOnFriday,
    #         #  'sat': usableOnSaturday,
    #         #  'sun': usableOnSunday,
    #         #  'ho': usableOnHoliday,
    #         #  'be': usableBeforeHoliday,
    #          'price': price if discount == 0 else discount,
    #          'ken': ken,
    #          'userKen': userKen,
    #         #  'ward': smallArea,
    #          'genre': genre,
    #         #  'type': capsule,
    #         #  'age': age,
    #          'sex': 1 if 'm' == sex else 0,
    #      }
    #      for
    #     #  listPeriod,
    #     #  validPeriod,
    #     #  usableOnMonday,
    #     #  usableOnTuesday,
    #     #  usableOnWednesday,
    #     #  usableOnThursday,
    #     #  usableOnFriday,
    #     #  usableOnSaturday,
    #     #  usableOnSunday,
    #     #  usableOnHoliday,
    #     #  usableBeforeHoliday,
    #      price,
    #      discount,
    #      userKen,
    #      ken,
    #     #  smallArea,
    #      genre,
    #     #  capsule,
    #     #  age,
    #      sex
    #      in features]
    # print('vectorizing...')
    # X = encoder.transform(X)
    # print('predicting...')
    # probList = clf.predict_proba(X)
    # print('ranking...')
    # labelTrueAt = np.where(clf.classes_ == 1)[0][0]
    # testDf['PURCHASE_PROB'] = probList[:, labelTrueAt]
    # testDf = testDf[['PURCHASE_PROB', 'DISCOUNT_PRICE', 'CATALOG_PRICE',
    #                  'USER_ID_hash', 'COUPON_ID_hash']]
    # testDf = testDf.sort_index(by=['PURCHASE_PROB',
    #                               'DISCOUNT_PRICE', 'CATALOG_PRICE'],
    #                           ascending=[False, False, True])
    # testDf = testDf[['USER_ID_hash', 'COUPON_ID_hash']]
    # topN = testDf.groupby(['USER_ID_hash']).apply(concat_top_n)
    # topN.name = 'PURCHASED_COUPONS'
    # topN.to_csv("submission_ft.csv", header=True)