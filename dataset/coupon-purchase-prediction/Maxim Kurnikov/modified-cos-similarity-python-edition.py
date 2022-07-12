__author__ = 'Maxim Kurnikov'

import pandas as pd
import numpy as np


coupon_list_train = pd.read_csv('../input/coupon_list_train.csv')
coupon_list_test = pd.read_csv('../input/coupon_list_test.csv')
user_list = pd.read_csv('../input/user_list.csv')
coupon_purchases_train = pd.read_csv("../input/coupon_detail_train.csv")

### merge to obtain (USER_ID) <-> (COUPON_ID with features) training set
purchased_coupons_train = coupon_purchases_train.merge(coupon_list_train,
                                                 on='COUPON_ID_hash',
                                                 how='inner')

### filter redundant features
features = ['COUPON_ID_hash', 'USER_ID_hash',
            'GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE',
            'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
            'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
            'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']
purchased_coupons_train = purchased_coupons_train[features]

### create 'dummyuser' records in order to merge training and testing sets in one
coupon_list_test['USER_ID_hash'] = 'dummyuser'

### filter testing set consistently with training set
coupon_list_test = coupon_list_test[features]

### merge sets together
combined = pd.concat([purchased_coupons_train, coupon_list_test], axis=0)

### create two new features
combined['DISCOUNT_PRICE'] = 1 / np.log10(combined['DISCOUNT_PRICE'])
combined['PRICE_RATE'] = (combined['PRICE_RATE'] / 100) ** 2
features.extend(['DISCOUNT_PRICE', 'PRICE_RATE'])

### convert categoricals to OneHotEncoder form
categoricals = ['GENRE_NAME', 'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED',
                'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN',
                'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']
combined_categoricals = combined[categoricals]
combined_categoricals = pd.get_dummies(combined_categoricals,
                                    dummy_na=False)

### leaving continuous features as is, obtain transformed dataset
continuous = list(set(features) - set(categoricals))
combined = pd.concat([combined[continuous], combined_categoricals], axis=1)

### remove NaN values
NAN_SUBSTITUTION_VALUE = 1
combined = combined.fillna(NAN_SUBSTITUTION_VALUE)

### split back into training and testing sets
train = combined[combined['USER_ID_hash'] != 'dummyuser']
test = combined[combined['USER_ID_hash'] == 'dummyuser']
test.drop('USER_ID_hash', inplace=True, axis=1)

### find most appropriate coupon for every user (mean of all purchased coupons), in other words, user profile
train_dropped_coupons = train.drop('COUPON_ID_hash', axis=1)
user_profiles = train_dropped_coupons.groupby(by='USER_ID_hash').apply(np.mean)

### creating weight matrix for features
FEATURE_WEIGHTS = {
    'GENRE_NAME': 2,
    'DISCOUNT_PRICE': 2,
    'PRICE_RATE': 0,
    'USABLE_DATE_': 0,
    'large_area_name': 0.5,
    'ken_name': 1,
    'small_area_name': 5
}

# dict lookup helper
def find_appropriate_weight(weights_dict, colname):
    for col, weight in weights_dict.items():
        if col in colname:
            return weight
    raise ValueError

W_values = [find_appropriate_weight(FEATURE_WEIGHTS, colname)
            for colname in user_profiles.columns]
W = np.diag(W_values)

### find weighted dot product(modified cosine similarity) between each test coupon and user profiles
test_only_features = test.drop('COUPON_ID_hash', axis=1)
similarity_scores = np.dot(np.dot(user_profiles, W),
                           test_only_features.T)

### create (USED_ID)x(COUPON_ID) dataframe, similarity scores as values
coupons_ids = test['COUPON_ID_hash']
index = user_profiles.index
columns = [coupons_ids[i] for i in range(0, similarity_scores.shape[1])]
result_df = pd.DataFrame(index=index, columns=columns,
                      data=similarity_scores)

### obtain string of top10 hashes according to similarity scores for every user
def get_top10_coupon_hashes_string(row):
    row.sort()
    return ' '.join(row.index[-10:][::-1].tolist())

output = result_df.apply(get_top10_coupon_hashes_string, axis=1)


output_df = pd.DataFrame(data={'USER_ID_hash': output.index,
                               'PURCHASED_COUPONS': output.values})
output_df_all_users = pd.merge(user_list, output_df, how='left', on='USER_ID_hash')
output_df_all_users.to_csv('cosine_sim_python.csv', header=True,
                           index=False, columns=['USER_ID_hash', 'PURCHASED_COUPONS'])














