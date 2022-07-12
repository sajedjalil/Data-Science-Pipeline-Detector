# Welcome to my kernel. This is designed as a simple kernel to make minimum preparations to
# the data and run it through a estimators which is very tolerant to messy data (LightGBM).
# My aim when starting a competition is to get on the score board as soon as possible and 
# build my way up.

# Credit to Wolfgang Beer who always puts out good simple kernels to learn from.

import numpy as np
import pandas as pd
import time
import gc
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

# A debug workflow to make a fast way to check everything is working
debug=False # False

if debug:
    nrows=10000 # In debug, bring only a small number of train and test rows
else:
    nrows=None

valid_fration=0.2 # The fraction of data for validation

# --------------------------- Data Loading --------------------------- 
t0 = time.time()

train = pd.read_csv('../input/train.csv', parse_dates=['activation_date'], nrows=nrows)
test = pd.read_csv('../input/test.csv', parse_dates=['activation_date'], nrows=nrows)

# Combine train and test so that feature creation is done simultaneously 
df = train.append(test, ignore_index=True)
train_len = len(train) # The length of the training data so we know where to split it later

# A little clean up to reduce memory usage
del train, test
gc.collect() # Garbage collection

print("Data loading done in (seconds):", time.time() - t0)

# --------------------------- Feature Engineering ---------------------------
t0 = time.time()

# A very simple data type adjustment for columns which could be categorical
for column in ['region', 'city', 'parent_category_name', 'category_name', 'user_type', 'image_top_1']:
    df[column] = df[column].astype('category')

# DateTime Features
df['weekday'] = df['activation_date'].dt.weekday.astype('uint8')

# Clean and Log price
df['price'] = df['price'].fillna(0)
df['price'] = np.log1p(df['price'])

# Groupby aggregations
s_user_posts = df.groupby(['user_id']).size().rename('user_posts')
df = df.join(s_user_posts, how='left', on='user_id')

# Word Count Features
def word_count(x):
    return len(str(x).split())
    
def unique_words(x):
    return len(set(str(x).split()))
    
def character_count(x):
    return len(str(x))

df['title_word_count'] = df['title'].apply(word_count)
df['desc_word_count'] = df['description'].apply(word_count)
df['title_unique_words'] = df['title'].apply(unique_words)
df['desc_unique_words'] = df['description'].apply(unique_words)
df['title_char_count'] = df['title'].apply(character_count)
df['desc_char_count'] = df['description'].apply(character_count)

# TF-IDF (Term Frequency - Inverse Document Frequency) + SVD
def VectDecomp(column, max_features=None): #Vectorize and Decompose
    t_VectDecomp = time.time()
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('russian'), max_features=max_features)
    tfidf_obj = vectorizer.fit_transform(column)
    svd_obj = TruncatedSVD(n_components=100, algorithm='arpack')
    df_svd = pd.DataFrame(svd_obj.fit_transform(tfidf_obj))
    df_svd.columns = ['svd_' + column.name + '_' + str(i+1) for i in range(100)]
    del tfidf_obj, vectorizer
    print("Completed VectDecomp for {} with max_features={} in {:.2f} seconds".format(column.name, max_features, time.time() - t_VectDecomp))
    return df_svd
    
title_svd = VectDecomp(df['title'])
df['description'].fillna('NA', inplace=True)
desc_svd = VectDecomp(df['description'], max_features=10000)
df = pd.concat([df, title_svd, desc_svd], axis=1)
svd_columns = title_svd.columns.tolist() + desc_svd.columns.tolist()
gc.collect()

print("Feature engineering done in (seconds):", time.time() - t0)
    
# -------------------------- Feature Selection & Splitting --------------------------- 
t0 = time.time()

# Select the column to use. Many removed now to keep it simple. Always remove the target column ('deal_probability')
selected_columns = [
 #'activation_date',
 'category_name',
 'city',
 #'deal_probability',
 #'description',
 #'image',
 'image_top_1',
 #'item_id',
 'item_seq_number',
 #'param_1',
 #'param_2',
 #'param_3',
 'parent_category_name',
 'price',
 'region',
 #'title',
 #'user_id',
 'user_type', 
 'weekday',
 'title_word_count',
 'desc_word_count',
 'title_unique_words',
 'desc_unique_words',
 'title_char_count',
 'desc_char_count',
# 'svd_title_1',
# 'svd_title_2',
# 'svd_title_3',
# 'svd_description_1',
# 'svd_description_2',
# 'svd_description_3',
 'user_posts',
    ]

selected_columns = selected_columns + svd_columns

# Split the data using the training data length from before
X_train, X_valid, y_train, y_valid = train_test_split(df.loc[:train_len-1, selected_columns],
                                                        df.loc[:train_len-1, ['deal_probability']],
                                                        test_size=valid_fration,
                                                        random_state=42
                                                        )
X_test = df.loc[train_len:, selected_columns]

print("Length X_train: {:,} and y_train: {:,}".format(X_train.shape[0], y_train.shape[0]))
print("Length X_valid: {:,} and y_valid: {:,}".format(X_valid.shape[0], y_valid.shape[0]))
print("Length X_test: {:,}".format(X_test.shape[0]))
print("Number of features: {}".format(len(selected_columns)))

print("Feature selection & spliting done in (seconds):", time.time() - t0)

# -------------------------- Model Training ---------------------------
t0 = time.time()

# Note: using Scikit-learn LightGBM API.
gbm = lgb.LGBMRegressor(random_state=12, learning_rate=0.2, n_estimators=300)
gbm.fit(X_train, y_train.values.ravel(),
        eval_set=[(X_valid, y_valid.values.ravel())],
        eval_metric='rmse', # The competition metric for scoring.
        early_stopping_rounds=5
       )

print("Training done in (seconds):", time.time() - t0)

# ----------------------- Feature Importance ----------------------
t0 = time.time()

# Find the important features and plot them in a chart (png file) found in the same folder.
feat_importances = pd.Series(gbm.feature_importances_, index=selected_columns)
feat_importances.nlargest(30).plot(kind='barh').invert_yaxis()
plt.savefig('feature_importances.png', dpi=100,bbox_inches="tight")

print("Feature importance check done in (seconds):", time.time() - t0)

# -------------------------- Prediction ---------------------------
t0 = time.time()

pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

print("Prediction done in (seconds):", time.time() - t0)

# ---------------------- Submission Creation -----------------------
t0 = time.time()

# Note: nrows here is case debuging is active. In which case, you cannot use 
# this submission as an entry.
submission = pd.read_csv('../input/sample_submission.csv', nrows=nrows, index_col='item_id') 
submission['deal_probability'] = pred
print("Prediction min {:+.3f} and max {:+.3f}".format(submission['deal_probability'].min(), submission['deal_probability'].max()))
submission['deal_probability'].clip(0.0, 1.0, inplace=True) # Temporary adjustement for predictions < 0. Investigate later.
submission.to_csv("submission.csv", columns=['deal_probability'], index=True,)

print("Submission creation done in (seconds):", time.time() - t0)