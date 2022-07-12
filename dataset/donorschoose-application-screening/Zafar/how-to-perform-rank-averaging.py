import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABEL = "project_is_approved"

predict_list = []
predict_list.append(pd.read_csv("../input/how-to-get-81-gru-att-lgbm-tf-idf-eda/submission.csv")[LABEL].values)
predict_list.append(pd.read_csv("../input/lightgbm-and-tf-idf-starter/submission.csv")[LABEL].values)
predict_list.append(pd.read_csv("../input/the-choice-is-yours/blend_submission.csv")[LABEL].values)
predict_list.append(pd.read_csv("../input/xtra-credit-xgb-w-tfidf-feature-stacking/blended_submission.csv")[LABEL].values)
predict_list.append(pd.read_csv("../input/keras-baseline-feature-hashing-price-tfidf/baseline_submission.csv")[LABEL].values)
predict_list.append(pd.read_csv("../input/lightgbm-and-nmf-starter-code/submission_nmf.csv")[LABEL].values)
predict_list.append(pd.read_csv("../input/a-pure-nlp-approach/submission.csv")[LABEL].values)


print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
        predictions = np.add(predictions, rankdata(predict)/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/donorschoose-application-screening/sample_submission.csv')
submission[LABEL] = predictions
submission.to_csv('rank_averaged_submission.csv', index=False)