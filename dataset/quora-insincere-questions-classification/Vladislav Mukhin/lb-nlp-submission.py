# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import string
from wordcloud import STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


def find_best_threshold_result(y_target, y_predicted, score=f1_score, thresholds=np.arange(.01, .81, .01)):
    threshold_scores = np.array([(threshold, score(y_target, y_predicted[:, 1] > threshold)) 
                                 for threshold in thresholds])
    best = threshold_scores[np.argmax(threshold_scores[:, 1])]
    return best


def print_results(y_target, y_predicted, score=f1_score, thresholds=np.arange(.01, .81, .01)):
    roc_auc = roc_auc_score(y_target, y_predicted[:, 1])
    best = find_best_threshold_result(y_target, y_predicted, score, thresholds)
    clf_report = classification_report(y_target, y_predicted[:, 1] > best[0])
    
    print("ROC_AUC: ", roc_auc)
    print("Threshold: ", best[0])
    print("Best score: ", best[1])
    print(clf_report)
    
    return best


train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")
test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

print("Train shape: ", train_df.shape)
print("Test shape: ", test_df.shape)

words_counter = lambda x: len(x.split()) 
unique_counter = lambda x: len(set(x.split()))
char_counter = len
stopwords_counter = lambda x: len([w for w in x.lower().split() if w in STOPWORDS])
punctuation_counter = lambda x: len([c for c in x if c in string.punctuation])
uppers_counter = lambda x: len([w for w in x.split() if w.isupper()])
title_words_counter = lambda x: len([w for w in x.split() if w.istitle()])
mean_word_len_counter = lambda x: np.mean([len(w) for w in x.split()])

train_df["n_words"] = train_df.question_text.map(words_counter)
train_df["n_words_unique"] = train_df.question_text.map(unique_counter)
train_df["n_chars"] = train_df.question_text.map(char_counter)
train_df["n_stopwords"] = train_df.question_text.map(stopwords_counter)
train_df["n_punctuations"] = train_df.question_text.map(punctuation_counter)
train_df["n_words_upper"] = train_df.question_text.map(uppers_counter)
train_df["n_words_title"] = train_df.question_text.map(title_words_counter)
train_df["mean_word_len"] = train_df.question_text.map(mean_word_len_counter)

test_df["n_words"] = test_df.question_text.map(words_counter)
test_df["n_words_unique"] = test_df.question_text.map(unique_counter)
test_df["n_chars"] = test_df.question_text.map(char_counter)
test_df["n_stopwords"] = test_df.question_text.map(stopwords_counter)
test_df["n_punctuations"] = test_df.question_text.map(punctuation_counter)
test_df["n_words_upper"] = test_df.question_text.map(uppers_counter)
test_df["n_words_title"] = test_df.question_text.map(title_words_counter)
test_df["mean_word_len"] = test_df.question_text.map(mean_word_len_counter)

additional_features = ["n_words", "n_words_unique", "n_chars", 
                       "n_stopwords", "n_punctuations", "n_words_upper", 
                       "n_words_title", "mean_word_len"]
F_train = train_df.loc[:, additional_features]
F_test = test_df.loc[:, additional_features]

scaler = StandardScaler()
scaler.fit(F_train)
F_train = scaler.transform(F_train)
F_test = scaler.transform(F_test)

y_train = train_df.target.values


vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.93, min_df=3)
X_train = vectorizer.fit_transform(train_df.question_text.values).astype(np.float32)
X_test = vectorizer.transform(test_df.question_text.values).astype(np.float32)

print(X_train.shape)
print(X_test.shape)

X_train = sparse.hstack((F_train, X_train), format='csr', dtype='float32')
X_test = sparse.hstack((F_test, X_test), format='csr', dtype='float32')

params = {
    "max_depth": 8,
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5, 
    "is_unbalance": True
}

n_splits = 3
cv = StratifiedKFold(n_splits, random_state=21, shuffle=True)

train_preds = np.zeros(train_df.shape[0])
boosters = []

for train_idx, valid_idx in cv.split(X_train, y_train):
    lgb_train = lgb.Dataset(X_train[train_idx], y_train[train_idx])
    lgb_valid = lgb.Dataset(X_train[valid_idx], y_train[valid_idx])
    
    gbm = lgb.train(params, lgb_train, 3000, valid_sets=[lgb_train, lgb_valid], 
                    early_stopping_rounds=100, verbose_eval=200)
    train_preds[valid_idx] = gbm.predict(X_train[valid_idx], num_iteration=gbm.best_iteration)
    boosters.append(gbm)
    
boosters_preds = [booster.predict(X_test, num_iteration=booster.best_iteration) for booster in boosters]

b_preds = np.average(boosters_preds, axis=0)
best_boosters = print_results(y_train, np.hstack([np.zeros((train_preds.shape[0], 1)), train_preds.reshape(-1, 1)]))
threshold = best_boosters[0]

predictions = (b_preds > threshold).astype(int)
submission = pd.DataFrame({"qid": test_df.qid.values, "prediction": predictions}, columns=["qid", "prediction"])
submission.to_csv("submission.csv", index=False)

