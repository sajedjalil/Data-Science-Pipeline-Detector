"""
This blend propelled me to my top score (0.9870) when used in conjunction with other blends so I am sharing it here in case it helps others too.
Even though I can't be sure this doesn't overfit, I think there is a good mix of models in there with low correlation. Possible improvements can
come from adding CNNs to this blend, but I haven't been able to get it to work well.

The models used were ran by Zafar (https://www.kaggle.com/fizzbuzz) on the Cleaned Toxic Comments dataset (https://www.kaggle.com/fizzbuzz/cleaned-toxic-comments).

Hope this helps!
"""



import numpy as np
import pandas as pd

gru_capsule = pd.read_csv('../input/capsule-net-with-gru-on-preprocessed-data/submission.csv')
gru_pool = pd.read_csv('../input/global-average-pool-on-preprocessed/submission.csv')
lstm_bi = pd.read_csv('../input/bilstm-on-preprocessed-data/submission.csv')
lstm_conv = pd.read_csv('../input/bi-lstm-conv-layer-lb-score-0-9840/submission.csv')
gru_fasttext = pd.read_csv('../input/pooled-gru-fasttext-on-preprocessed-data/submission.csv')
ridge = pd.read_csv('../input/ridge-on-words-char-n-gram-preprocessed-data/submission.csv')



from sklearn.preprocessing import minmax_scale
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in labels:
    gru_capsule[label] = minmax_scale(gru_capsule[label])
    gru_pool[label] = minmax_scale(gru_pool[label])
    lstm_bi[label] = minmax_scale(lstm_bi[label])
    lstm_conv[label] = minmax_scale(lstm_conv[label])
    gru_fasttext[label] = minmax_scale(gru_fasttext[label])
    ridge[label] = minmax_scale(ridge[label])



submission = pd.DataFrame()
submission['id'] = gru_capsule['id']

submission[labels] = (gru_capsule[labels]*1 + \
                     gru_pool[labels]*1 + \
                     lstm_bi[labels]*1 + \
                     lstm_conv[labels]*1 + \
                     gru_fasttext[labels]*1 + \
                     ridge[labels]*1) / 6


submission.to_csv('preprocessed_blend.csv', index=False)