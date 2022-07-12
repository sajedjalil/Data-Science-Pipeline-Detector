import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import lightgbm as lgb
import xgboost as xgb

import gc

NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000

print("Reading in Data")

df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')

df = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]
print(nrow_train)
y_train = np.log1p(df_train["price"])
y = np.log1p(df_train["price"])

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

print(X_train.shape, X_test.shape)

train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size = 0.1, random_state = 144) 

dtrain = xgb.DMatrix(train_X, label=train_y)
dvalid = xgb.DMatrix(valid_X, label=valid_y)
xX_test = xgb.DMatrix(X_test)
#dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

params = {'min_child_weight': 20, 'eta': 0.013, 'colsample_bytree': 0.45, 'max_depth': 16,
            'subsample': 0.88, 'lambda': 2.07, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

#model = xgb.train(params, dtrain, 300, watchlist, verbose_eval=10, early_stopping_rounds=20)
model = xgb.train(params, dtrain, 300, watchlist, early_stopping_rounds=20, verbose_eval=10)
    
preds = model.predict(xX_test) #, ntree_limit=model.best_ntree_limit)

df_test["price"] = np.expm1(preds)
df_test[["test_id", "price"]].to_csv("submission_XGB.csv", index = False) # 0.54007

model = Ridge(solver="saga", fit_intercept=True, random_state=205)
model.fit(X_train, y)
# print('[{}] Finished to train ridge'.format(time.time() - start_time))
pred = model.predict(X=X_test)
# print('[{}] Finished to predict ridge'.format(time.time() - start_time))

df_test["price"] = np.expm1(pred)
df_test[["test_id", "price"]].to_csv("submission_Ridge.csv", index = False) # 0.46574

preds += pred
preds /=2
df_test["price"] = np.expm1(preds)
df_test[["test_id", "price"]].to_csv("submission_XGB_Ridge.csv", index = False) # 0.47508