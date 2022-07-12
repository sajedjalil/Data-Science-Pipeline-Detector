# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import scipy
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge, LogisticRegression, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import SGDRegressor
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


import gc

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000

print("Reading in Data")
def RSMLE(Y, Y_pred):
    return np.sqrt(np.mean(np.square(Y-Y_pred)))
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')
df_train = df_train.drop(df_train[(df_train.price < 3.0)].index)

df_train['subcat_0'], df_train['subcat_1'], df_train['subcat_2'] = \
zip(*df_train['category_name'].apply(lambda x: split_cat(x)))
df_test['subcat_0'], df_test['subcat_1'], df_test['subcat_2'] = \
zip(*df_test['category_name'].apply(lambda x: split_cat(x)))

def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
df_train, df_val = train_test_split(df_train, random_state=123, train_size=0.99)
nrow_train = df_train.shape[0]
nrow_val =df_val.shape[0]
df = pd.concat([df_train, df_val, df_test], 0)

y_train = np.log1p(df_train["price"])
y_val = np.log1p(df_val["price"])


del df_train
gc.collect()

print(df.memory_usage(deep = True))
# get name and description lengths
def wordCount(text):
    try:
        if text == 'No description yet':
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except: 
        return 0
df['desc_len'] = df['item_description'].apply(lambda x: wordCount(x))
df['name_len'] = df['name'].apply(lambda x: wordCount(x))
df["category_name"] = df["category_name"].fillna("missing").astype("category")
df["brand_name"] = df["brand_name"].fillna("missing").apply(lambda x: x.lower())

cutting(df)

df["item_description"] = df["item_description"].fillna("None").apply(lambda x: x.lower())
df['category_name'] = df['category_name'].fillna('missing').astype(str)
df['subcat_0'] = df['subcat_0'].astype(str)
df['subcat_1'] = df['subcat_1'].astype(str)
df['subcat_2'] = df['subcat_2'].astype(str)
df['brand_name'] = df['brand_name'].fillna('missing').astype(str)
df['shipping'] = df['shipping'].astype(str)
df['item_condition_id'] = df['item_condition_id'].astype(str)
df['desc_len'] = df['desc_len'].astype(str)
df['name_len'] = df['name_len'].astype(str)
df['item_description'] = df['item_description'].fillna('No description yet').astype(str)
df['trainrow'] = np.arange(df.shape[0])
print(df.memory_usage(deep = True)) 
print("Vectorizing data...")
default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('subcat_0', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_0'))),
    ('subcat_1', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_1'))),
    ('subcat_2', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_2'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('desc_len', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('desc_len'))),
    ('name_len', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('name_len'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        preprocessor=build_preprocessor('item_description'))),
])

X = vectorizer.fit_transform(df.values)

X_train = X[:nrow_train]

X_val = X[nrow_train:nrow_train+nrow_val]

X_test = X[nrow_train+nrow_val:]
print(X.shape, X_train.shape, X_val.shape, X_test.shape)
print("Fitting Model")
X_test = X[nrow_train+nrow_val:]
#model
d_train = lgb.Dataset(X_train, label=y_train)
watchlist = [d_train]
params = {
        'learning_rate': 0.6,
        'application': 'regression',
        'max_depth': 5,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.6,
        'nthread': 4,
        'min_data_in_leaf': 100,
        'max_bin': 31
        }

model2 = lgb.train(params, train_set=d_train, num_boost_round=5200, valid_sets=watchlist, \
                      early_stopping_rounds=1000, verbose_eval=1000) 
ridge_modelCV = RidgeCV(
    fit_intercept=True, alphas=[5.0],
    normalize=False, cv = 2, scoring='neg_mean_squared_error',
)
ridge_modelCV.fit(X_train, y_train)
print("Fitting Model")
pred_val = 0.5*model2.predict(X_val)+0.5*ridge_modelCV.predict(X_val)
print("predicting")
print(RSMLE(y_val, pred_val))
preds = 0.5*model2.predict(X_test)+0.5*ridge_modelCV.predict(X_test)
df_test["price"] = np.expm1(preds)
df_test[["test_id", "price"]].to_csv("sample_submission.csv", index = False)            