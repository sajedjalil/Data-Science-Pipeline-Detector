import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import xgboost as xgb
from xgboost.sklearn import XGBClassifier # <3
from sklearn.model_selection import train_test_split
import gc

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')#.sample(1000)
test = pd.read_csv('../input/test.csv').fillna(' ')#.sample(1000)

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

train = train.loc[:,class_names]

print("TFIDF")
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    norm='l2',
    min_df=0,
    smooth_idf=False,
    max_features=15000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    norm='l2',
    min_df=0,
    smooth_idf=False,
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
del train_char_features,train_word_features
test_features = hstack([test_char_features, test_word_features])
del test_char_features,test_word_features

print(train_features.shape)
print(test_features.shape)
d_test = xgb.DMatrix(test_features)
del test_features
gc.collect()

print("Modeling")
cv_scores = []
xgb_preds = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    # Split out a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_features, train_target, test_size=0.25, random_state=23)

    xgb_params = {'eta': 0.3, 
              'max_depth': 5, 
              'subsample': 0.8, 
              'colsample_bytree': 0.8, 
              'objective': 'binary:logistic', 
              'eval_metric': 'auc', 
              'seed': 23
             }

    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 200, watchlist, verbose_eval=False, early_stopping_rounds=30)
    print("class Name: {}".format(class_name))
    print(model.attributes()['best_msg'])
    cv_scores.append(float(model.attributes()['best_score']))
    submission[class_name] = model.predict(d_test)
    del X_train, X_valid, y_train, y_valid
    gc.collect()
print('Total CV score is {}'.format(np.mean(cv_scores)))
submission.to_csv('submission.csv', index=False)