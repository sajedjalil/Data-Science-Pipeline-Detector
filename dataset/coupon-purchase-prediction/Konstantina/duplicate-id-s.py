import pandas as pd

coupons_area_train = pd.read_csv('../input/coupon_area_train.csv')
coupons_area_train = coupons_area_train.groupby('COUPON_ID_hash')
print(coupons_area_train.count())

print(coupons_area_train.groupby(['COUPON_ID_hash', 'PREF_NAME']).size())