import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

predict_list = []
predict_list.append(pd.read_csv("../input/textcnn-2d-convolution-on-preprocessed-data/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/lr-with-words-and-char-n-grams-preprocessed-data/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/pooled-gru-fasttext-on-preprocessed-data/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/cnn-3-out-of-fold-4-epochs-preprocessed-data/submit_cnn_avg_3_folds.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/global-average-pool-on-preprocessed/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/lemmatization-pooled-gru-on-preprocessed-dataset/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/bilstm-on-preprocessed-data/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/wordbatch-fm-ftrl-on-preprocessed-data/lvl0_wordbatch_clean_sub.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/ridge-on-words-char-n-gram-preprocessed-data/submission.csv")[LABELS].values)


print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(6):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[LABELS] = predictions
submission.to_csv('rank_averaged_submission.csv', index=False)