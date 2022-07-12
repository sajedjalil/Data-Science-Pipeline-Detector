import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

predict_list = []

# https://www.kaggle.com/rednivrug/blend-it-all
predict_list.append(pd.read_csv("../input/blend-it-all/blend_it_all.csv")[LABELS].values)

# https://www.kaggle.com/the1owl/toxic-simple-blending-toxic-avenger-spin
predict_list.append(pd.read_csv("../input/toxic-simple-blending-toxic-avenger-spin/submission.csv")[LABELS].values)

# https://www.kaggle.com/prashantkikani/toxic-one-more-blend
predict_list.append(pd.read_csv("../input/toxic-one-more-blend/one_more_blend.csv")[LABELS].values)

# https://www.kaggle.com/prashantkikani/hight-of-blend-v2
predict_list.append(pd.read_csv("../input/hight-of-blend-v2/hight_of_blend_v2.csv")[LABELS].values)

# https://www.kaggle.com/hhstrand/oof-stacking-regime
predict_list.append(pd.read_csv("../input/oof-stacking-regime/submission.csv")[LABELS].values)

# https://www.kaggle.com/fizzbuzz/rank-averaging-on-preprocessed-data
predict_list.append(pd.read_csv("../input/rank-averaging-on-preprocessed-data/rank_averaged_submission.csv")[LABELS].values)

# https://www.kaggle.com/antmarakis/another-blend-tinkered-by-correlation
predict_list.append(pd.read_csv("../input/another-blend-tinkered-by-correlation/corr_blend.csv")[LABELS].values)

# https://www.kaggle.com/tunguz/blend-of-blends-1
predict_list.append(pd.read_csv("../input/blend-of-blends-1/superblend_1.csv")[LABELS].values)

# https://www.kaggle.com/peterhurford/lgb-gru-lr-lstm-nb-svm-average-ensemble
predict_list.append(pd.read_csv("../input/lgb-gru-lr-lstm-nb-svm-average-ensemble/submission.csv")[LABELS].values)

# https://www.kaggle.com/jetouxu/toxic-one-more-b8bce2
predict_list.append(pd.read_csv("../input/toxic-one-more-b8bce2/one_more_blend.csv")[LABELS].values)

predict_list.append(pd.read_csv("../input/toxic-ridge/10-fold_elast_test.csv")[LABELS].values)

#predict_list.append(pd.read_csv("../input/toxic-ridge/10-fold_ridge_train.csv")[LABELS].values)


print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(6):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[LABELS] = predictions
submission.to_csv('superblend.csv', index=False)