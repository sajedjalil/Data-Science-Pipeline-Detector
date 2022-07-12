"""
This blend combines some high-rated public kernels with low(-ish) correlation.
"""


import numpy as np
import pandas as pd

lstm = pd.read_csv('../input/bidirectional-lstm-with-convolution/submission.csv') #0.9841
lr = pd.read_csv('../input/tuned-logreg-oof-files/submission-tuned-LR-01.csv') #0.9800
gru_cnn = pd.read_csv('../input/bi-gru-cnn-poolings/submission.csv') #0.9841
r_script = pd.read_csv('../input/why-a-such-low-score-with-r-and-keras/submission.csv') #0.9824



from sklearn.preprocessing import minmax_scale
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in labels:
    r_script[label] = minmax_scale(r_script[label])
    lstm[label] = minmax_scale(lstm[label])
    lr[label] = minmax_scale(lr[label])
    gru_cnn[label] = minmax_scale(gru_cnn[label])



submission = pd.DataFrame()
submission['id'] = r_script['id']

submission['toxic'] = r_script['toxic'] * 0.25 + \
                      lstm['toxic'] * 0.3 + \
                      lr['toxic'] * 0.2 + \
                      gru_cnn['toxic'] * 0.25

submission['severe_toxic'] = r_script['severe_toxic'] * 0.25 + \
                             lstm['severe_toxic'] * 0.3 + \
                             lr['severe_toxic'] * 0.2 + \
                             gru_cnn['severe_toxic'] * 0.25

submission['obscene'] = r_script['obscene'] * 0.2 + \
                        lstm['obscene'] * 0.3 + \
                        lr['obscene'] * 0.2 + \
                        gru_cnn['obscene'] * 0.3

submission['threat'] = r_script['threat'] * 0.25 + \
                       lstm['threat'] * 0.25 + \
                       lr['threat'] * 0.2 + \
                       gru_cnn['threat'] * 0.3

submission['insult'] = r_script['insult'] * 0.25 + \
                       lstm['insult'] * 0.3 + \
                       lr['insult'] * 0.2 + \
                       gru_cnn['insult'] * 0.25

submission['identity_hate'] = r_script['identity_hate'] * 0.25 + \
                              lstm['identity_hate'] * 0.3 + \
                              lr['identity_hate'] * 0.2 + \
                              gru_cnn['identity_hate'] * 0.25

submission.to_csv('corr_blend.csv', index=False)