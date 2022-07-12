"""
LGBM Regression on TfIDF of text features and One-Hot-Encoded Categoricals
Featues based on Alexandu Papiu's (https://www.kaggle.com/apapiu) script: https://www.kaggle.com/apapiu/ridge-script
LGBM based on InfiniteWing's (https://www.kaggle.com/infinitewing) script: https://www.kaggle.com/infinitewing/lightgbm-example
"""

import pandas as pd
import numpy as np
import scipy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import lightgbm as lgb

import gc

NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000

print("Reading in Data")

df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')

df = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]
y_train = np.log1p(df_train["price"])

del df_train
gc.collect()

print(df.memory_usage(deep = True))

df["category_name"] = df["category_name"].fillna("Other").astype("category")
df["brand_name"] = df["brand_name"].fillna("unknown")

pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

df["item_description"] = df["item_description"].fillna("None")
df["item_condition_id"] = df["item_condition_id"].astype("category")
df["brand_name"] = df["brand_name"].astype("category")

print(df.memory_usage(deep = True))

print("Encodings")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(df["name"])

print("Category Encoders")
unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
count_category = CountVectorizer()
X_category = count_category.fit_transform(df["category_name"])

print("Descp encoders")
count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                              ngram_range = (1,3),
                              stop_words = "english")
X_descp = count_descp.fit_transform(df["item_description"])

print("Brand encoders")
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(df["brand_name"])

print("Dummy Encoders")
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
    "item_condition_id", "shipping"]], sparse = True).values)

X = scipy.sparse.hstack((X_dummies, 
                         X_descp,
                         X_brand,
                         X_category,
                         X_name)).tocsr()

print([X_dummies.shape, X_category.shape, 
       X_name.shape, X_descp.shape, X_brand.shape])

X_train = X[:nrow_train]
X_test = X[nrow_train:]



params = {
    'learning_rate': 0.75,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 100,
    'verbosity': -1,
    'metric': 'RMSE',
}


train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size = 0.1, random_state = 144) 
d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
watchlist = [d_train, d_valid]

model = lgb.train(params, train_set=d_train, num_boost_round=2200, valid_sets=watchlist, \
early_stopping_rounds=50, verbose_eval=100) 
preds = model.predict(X_test)

model = Ridge(solver = "lsqr", fit_intercept=False)

print("Fitting Model")
# model.fit(X_train, y_train)

# preds += model.predict(X_test)
# preds /= 2


df_test["price"] = np.expm1(preds)
df_test[["test_id", "price"]].to_csv("submission_LGBM_Ridge_3.csv", index = False)