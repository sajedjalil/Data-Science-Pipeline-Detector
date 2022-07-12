#! /usr/bin/env python3.4

import pandas as pd
import numpy as np


def concat_top_n(df, n=10):
    return ' '.join(df[:n]['COUPON_ID_hash'])


if __name__ == '__main__':
    couponListTestDf = pd.read_csv('../input/coupon_list_test.csv')
    userListDf = pd.read_csv('../input/user_list.csv')[['USER_ID_hash',
                                                        'PREF_NAME']].fillna('東京都')
    kenCouponListDf = pd.merge(userListDf, couponListTestDf,
                              left_on='PREF_NAME', right_on='ken_name',
                              how='left').fillna('N/A')
    kenCouponListDf = kenCouponListDf[['USER_ID_hash', 'COUPON_ID_hash']]
    topN = kenCouponListDf.groupby('USER_ID_hash').apply(concat_top_n)
    topN.name = 'PURCHASED_COUPONS'
    topN.to_csv("submission_same-ken.csv", header=True)
