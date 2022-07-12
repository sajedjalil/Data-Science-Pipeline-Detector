"""
Ridge Script (kudos to https://www.kaggle.com/apapiu/ridge-script)
extended with Truncated SVD, and added validation set
"""

import pandas as pd
import numpy as np
import scipy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import TruncatedSVD
import math

import gc


NUM_BRANDS = 6500
NAME_MIN_DF = 2
MAX_FEAT_DESCP = 100000


# Definitions
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5


print("Reading in Data")

df_trains = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')

df_train=df_trains.sample(frac=0.9,random_state=200)
df_validation=df_trains.drop(df_train.index)
del(df_trains)

df = pd.concat([df_train,df_validation, df_test], 0)
nrow_train = df_train.shape[0]
nrow_validation = df_validation.shape[0]
y_train = np.log1p(df_train["price"])
y_validation = np.log1p(df_validation["price"])

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

del(df)
gc.collect()

from datetime import datetime
start=datetime.now()

print('TSVD')
tsvd = TruncatedSVD(n_iter=3, n_components=90)
truncated_item_description = tsvd.fit_transform(X_descp)

#tsvd = TruncatedSVD(n_iter=1, n_components=20)
#truncated_name = tsvd.fit_transform(X_name)
del(tsvd)
gc.collect()

stop=datetime.now()
print('TSVD took ', stop-start)


X = scipy.sparse.hstack((X_dummies, 
                         X_descp,
                         truncated_item_description,
                         #truncated_name,
                         X_brand,
                         X_category,
                         X_name)).tocsr()

print([X_dummies.shape, X_category.shape, 
       X_name.shape, X_descp.shape, X_brand.shape])

X_train = X[:nrow_train]
model = Ridge(alpha=0.8, solver = "lsqr", fit_intercept=False)

print("Fitting Model")
model.fit(X_train, y_train)

print('Predicting validation set')
X_validation = X[nrow_train:(nrow_train+nrow_validation)]
preds_validation = model.predict(X_validation)
print('RMSLE on validation set',
    rmsle(np.expm1(np.asarray(y_validation)), np.expm1(np.abs(preds_validation)) ))

print('Predicting test set')
X_test = X[(nrow_train+nrow_validation):]
preds = model.predict(X_test)

df_test["price"] = np.expm1(preds)
df_test[["test_id", "price"]].to_csv("submission_ridge_tsvd.csv", index = False)