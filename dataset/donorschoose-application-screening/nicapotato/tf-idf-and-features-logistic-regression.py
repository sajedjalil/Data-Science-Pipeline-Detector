import time
start = time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import gc
import os
print("Files:", os.listdir("../input"))

# Warnings
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("../input/train.csv",index_col="id",low_memory=False,
                    parse_dates=["project_submitted_datetime"])#.sample(1000)
print("Train Data Shape: ",train.shape)
train = train.sample(115000,random_state = 23)
traindex = train.index
test = pd.read_csv("../input/test.csv",index_col="id",low_memory=False,
                    parse_dates=["project_submitted_datetime"])#.sample(1000)
tesdex = test.index
project_is_approved = train["project_is_approved"].copy()
df = pd.concat([train.drop("project_is_approved",axis=1),test],axis=0).drop("teacher_id",axis=1)
rc = pd.read_csv("../input/resources.csv",index_col="id").fillna("missingpotato")

print("Merge..")
# Aggregate and Merge
agg_rc = rc.reset_index().groupby('id').agg(dict(quantity = 'sum', price = 'sum', description = lambda x: ' nicapotato '.join(x)))
df = pd.merge(df,agg_rc, left_index=True, right_index=True, how= "inner")
alldex = df.index
del test, train, rc,agg_rc

print("Creating Features..")

"""
Feature Engineering
Here I combine all sources of text into one.
Purposed of this is to use the TF IDF transformation all at once.
"""

df['text'] = df.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    str(row['project_resource_summary']),
    str(row['project_title']),
    str(row['description'])]), axis=1)
df = pd.merge(df, df["project_subject_categories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="inner")
df = pd.merge(df, df["project_subject_subcategories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="inner")

all_text = df['text']
df.drop('text', axis=1, inplace=True)

# Time Frames of Interest
df["Year"] = df["project_submitted_datetime"].dt.year
df["Date of Year"] = df['project_submitted_datetime'].dt.dayofyear # Day of Year
df["Weekday"] = df['project_submitted_datetime'].dt.weekday
df["Day of Month"] = df['project_submitted_datetime'].dt.day

print("Dummies..")
# Dummies - This notebook revolves around making my data sparse.
df = pd.get_dummies(df, columns=[
    'Weekday','Day of Month','Year','Date of Year',
    'teacher_prefix','school_state','project_grade_category'#,"teacher_id"
    ],
    sparse=True)

df.drop(['project_essay_1','project_essay_2','project_essay_3','project_essay_4'
         ,'project_subject_categories',"project_subject_subcategories",
        "project_resource_summary","project_title","description","project_submitted_datetime"],axis=1,inplace=True)
normalize = ["teacher_number_of_previously_posted_projects","quantity","price"]

# Standardize
"""
Perhaps try to leave these as bools at some point -
"""
std = StandardScaler()
# for col in normalize:
#     df[col] = std.fit_transform(df[col].values.reshape(-1, 1))

print("Creating Word Features Matrix..")
# build TFIDF Vectorizer
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    dtype=np.float32,
    max_features=5000
)


# Character Stemmer
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    dtype=np.float32,
    max_features=4000
)

word_vectorizer.fit(all_text[traindex])
char_vectorizer.fit(all_text[traindex])

# Train

train_word_features = word_vectorizer.transform(all_text[traindex])
train_char_features = char_vectorizer.transform(all_text[traindex])

normdf = pd.DataFrame(std.fit_transform(df),columns=df.columns).set_index(alldex)
del df

print("Sparse Matrix..")
# Sparse Matrix
train_features = hstack([
    train_char_features,
    train_word_features
    ,csr_matrix(normdf.loc[traindex,])], 'csr'
)
del train_word_features, train_char_features
print("train shape: {} rows, {}".format(*train_features.shape))

# Test
test_word_features = word_vectorizer.transform(all_text[tesdex])
test_char_features = char_vectorizer.transform(all_text[tesdex])
del word_vectorizer, char_vectorizer, std, all_text
gc.collect()

test_features = hstack([
    test_char_features,
    test_word_features
    ,csr_matrix(normdf.loc[tesdex,])], 'csr'
)
del test_word_features, test_char_features
gc.collect()

print("test shape: {} rows, {}".format(*test_features.shape))

# Model
print("Modeling..")
loss = []
lr = LogisticRegression(solver="sag", max_iter=200)
lr.fit(train_features,project_is_approved)
print("Auc Score: ",np.mean(cross_val_score(lr, train_features, project_is_approved, cv=3, scoring='roc_auc')))

sub = pd.DataFrame(lr.predict_proba(test_features)[:, 1],columns=["project_is_approved"],index=tesdex)
sub.to_csv("logistic_sub.csv",index=True)
print("Notebook took %0.2f minutes to Run"%((time.time() - start)/60))