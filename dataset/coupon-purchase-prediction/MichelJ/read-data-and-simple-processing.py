### Kaggle Scripts: Ponpare Coupon Purchase Prediction ###
### Author: MichelJ ###

# Read the data files, using Pandas, and do some simple processing

import os
import math
import pandas as pd
import numpy as np

os.system("ls ../input")
# os.system("echo \n\n")
# os.system("head ../input/*")

#read in all the input data
cpdtr = pd.read_csv("../input/coupon_detail_train.csv")
cpltr = pd.read_csv("../input/coupon_list_train.csv")
cplte = pd.read_csv("../input/coupon_list_test.csv")
ulist = pd.read_csv("../input/user_list.csv")
# areas = pd.read_csv("../input/prefecture_locations.csv")

####### BEGIN Translate capsule text and genre name
# __author__ = 'Toby Cheese'
# read the file and parse the first and only sheet (need python xlrd module)
f = pd.ExcelFile('../input/documentation/CAPSULE_TEXT_Translation.xlsx')
all = f.parse(parse_cols=[2,3,6,7], skiprows=4, header=1)

# data comes in two columns, produce a single lookup table from that
first_col = all[['CAPSULE_TEXT', 'English Translation']]
second_col = all[['CAPSULE_TEXT.1','English Translation.1']].dropna()
second_col.columns = ['CAPSULE_TEXT', 'English Translation']
all = first_col.append(second_col).drop_duplicates('CAPSULE_TEXT')
translation_map = {k:v for (k,v) in zip(all['CAPSULE_TEXT'], all['English Translation'])}
####### END Translate capsule text and genre name

#making of the train set
train = pd.merge(cpdtr, cpltr)
train = train[["COUPON_ID_hash","USER_ID_hash",
                  "GENRE_NAME","DISCOUNT_PRICE","PRICE_RATE",
                  "USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU",
                  "USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY",
                  "USABLE_DATE_BEFORE_HOLIDAY","ken_name","small_area_name"]]

#combine the test set with the train
cplte['USER_ID_hash'] = "dummyuser"
cpchar = cplte[["COUPON_ID_hash","USER_ID_hash",
                   "GENRE_NAME","DISCOUNT_PRICE","PRICE_RATE",
                   "USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU",
                   "USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY",
                   "USABLE_DATE_BEFORE_HOLIDAY","ken_name","small_area_name"]]

train.append(cpchar)

# Translation
# train['CAPSULE_TEXT'] = train['CAPSULE_TEXT'].map(translation_map)
train['GENRE_NAME'] = train['GENRE_NAME'].map(translation_map)

#NA imputation
train = train.fillna(1)

# print(train['DISCOUNT_PRICE'])

#feature engineering

#train['DISCOUNT_PRICE'] = 1 / math.log10(1.0 * train['DISCOUNT_PRICE'])
train['DISCOUNT_PRICE'] = train['DISCOUNT_PRICE'].apply(lambda x : 1 / math.log10(x + 0.001))
#train['PRICE_RATE'] = train['PRICE_RATE'] * train['PRICE_RATE'] / 10000
train['PRICE_RATE'] = train['PRICE_RATE'].apply(lambda x : x * x / 10000)

print(train.shape)
print(train.dtypes)
print(train[["COUPON_ID_hash","USER_ID_hash",
                  "GENRE_NAME","DISCOUNT_PRICE","PRICE_RATE",
                  "USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU",
                  "USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY",
                  "USABLE_DATE_BEFORE_HOLIDAY"]].describe())
