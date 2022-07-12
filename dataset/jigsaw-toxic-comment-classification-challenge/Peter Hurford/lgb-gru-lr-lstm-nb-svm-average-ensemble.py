# Playing with submitting an average of high scoring kernels, as it seems
# everyone else is doing. Weights were determined by guessing based on
# their relative public leaderboard scores.
#
# This probably overfits, and proper work would involve running all these
# kernels with CV and determining the proper weights on the OOF predictions.

import numpy as np
import pandas as pd


gru = pd.read_csv('../input/pooled-gru-fasttext/submission.csv') # PL score 0.9829
lstm_nb_svm = pd.read_csv('../input/minimal-lstm-nb-svm-baseline-ensemble/submission.csv') # 0.9811
lr = pd.read_csv('../input/logistic-regression-with-words-and-char-n-grams/submission.csv') # 0.9788
lgb = pd.read_csv('../input/lightgbm-with-select-k-best-on-tfidf/lgb_submission.csv') # 0.9785


# The value of an ensemble is (a) the individual scores of the models and
# (b) their correlation with one another. We want multiple individually high
# scoring models that all have low correlations. Based on this analysis, it
# looks like these kernels have relatively low correlations and will blend to a
# much higher score.
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in labels:
    print(label)
    print(np.corrcoef([lgb[label].rank(pct=True), gru[label].rank(pct=True), lr[label].rank(pct=True), lstm_nb_svm[label].rank(pct=True)]))

submission = pd.DataFrame()
submission['id'] = lgb['id']
for label in labels:
    submission[label] = lgb[label].rank(pct=True) * 0.15 + gru[label].rank(pct=True) * 0.4 + lr[label].rank(pct=True) * 0.15 + lstm_nb_svm[label].rank(pct=True) * 0.3

submission.to_csv('submission.csv', index=False)