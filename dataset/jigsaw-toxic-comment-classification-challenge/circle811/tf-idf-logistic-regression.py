import os
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def predict_one(x, y, xt):
    c = LogisticRegression(C=1.2, class_weight='balanced')
    c.fit(x, y)
    y_pred = c.predict_proba(xt)
    idx = list(c.classes_).index(1)
    return y_pred[:, idx]


print(os.listdir('../input'))

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
print('load data')

all_comment_text = pd.concat([train_data.comment_text, test_data.comment_text])

char_vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), max_features=50000)
char_vec.fit(all_comment_text)
x_train_c = char_vec.transform(train_data.comment_text)
x_test_c = char_vec.transform(test_data.comment_text)

word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=50000)
word_vec.fit(all_comment_text)
x_train_w = word_vec.transform(train_data.comment_text)
x_test_w = word_vec.transform(test_data.comment_text)

x_train_j = sp.sparse.hstack([x_train_c, x_train_w])
x_test_j = sp.sparse.hstack([x_test_c, x_test_w])
print('prepare data')

sub_c = pd.DataFrame({'id': test_data['id']})
sub_w = pd.DataFrame({'id': test_data['id']})
sub_j = pd.DataFrame({'id': test_data['id']})
sub_mean = pd.DataFrame({'id': test_data['id']})
sub_max = pd.DataFrame({'id': test_data['id']})

targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for t in targets:
    y = train_data[t].values
    y_pred_c = predict_one(x_train_c, y, x_test_c)
    y_pred_w = predict_one(x_train_w, y, x_test_w)
    y_pred_j = predict_one(x_train_j, y, x_test_j)
    sub_c[t] = y_pred_c
    sub_w[t] = y_pred_w
    sub_j[t] = y_pred_j
    sub_mean[t] = 0.5 * (y_pred_c + y_pred_w)
    sub_max[t] = np.maximum(y_pred_c, y_pred_w)
    print('predict {}'.format(t))

sub_c.to_csv('char.csv', index=0)
sub_w.to_csv('word.csv', index=0)
sub_j.to_csv('joint.csv', index=0)
sub_mean.to_csv('mean.csv', index=0)
sub_max.to_csv('max.csv', index=0)
