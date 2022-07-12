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


# Bojan suggests scaling with min-max to make sure that all the submissions have
# orderings that can be compared. Since our metric is AUC, this is okay and may
# improve performance.
from sklearn.preprocessing import minmax_scale
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in labels:
    print('Scaling {}... Please stand by.'.format(label))
    lgb[label] = minmax_scale(lgb[label])
    gru[label] = minmax_scale(gru[label])
    lr[label] = minmax_scale(lr[label])
    lstm_nb_svm[label] = minmax_scale(lstm_nb_svm[label])


# The value of an ensemble is (a) the individual scores of the models and
# (b) their correlation with one another. We want multiple individually high
# scoring models that all have low correlations. Based on this analysis, it
# looks like these kernels have relatively low correlations and will blend to a
# much higher score.
for label in labels:
    print(label)
    print(np.corrcoef([lgb[label], gru[label], lr[label], lstm_nb_svm[label]]))
    

#submission = pd.DataFrame()
#submission['id'] = lgb['id']
#submission['toxic'] = lgb['toxic'] * 0.15 + gru['toxic'] * 0.4 + lr['toxic'] * 0.15 + lstm_nb_svm['toxic'] * 0.3
#submission['severe_toxic'] = lgb['severe_toxic'] * 0.15 + gru['severe_toxic'] * 0.4 + lr['severe_toxic'] * 0.15 + lstm_nb_svm['severe_toxic'] * 0.3
#submission['obscene'] = lgb['obscene'] * 0.15 + gru['obscene'] * 0.4 + lr['obscene'] * 0.15 + lstm_nb_svm['obscene'] * 0.3
#submission['threat'] = lgb['threat'] * 0.15 + gru['threat'] * 0.4 + lr['threat'] * 0.15 + lstm_nb_svm['threat'] * 0.3
#submission['insult'] = lgb['insult'] * 0.15 + gru['insult'] * 0.4 + lr['insult'] * 0.15 + lstm_nb_svm['insult'] * 0.3
#submission['identity_hate'] = lgb['identity_hate'] * 0.15 + gru['identity_hate'] * 0.4 + lr['identity_hate'] * 0.15 + lstm_nb_svm['identity_hate'] * 0.3
#submission.to_csv('submission.csv', index=False)
