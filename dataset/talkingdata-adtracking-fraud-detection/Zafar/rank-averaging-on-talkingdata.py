import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABEL = "is_attributed"

predict_list = []

for i in range(1,10):
    predict_list.append(pd.read_csv("../input/kernel-%03d/submission.csv"%i)[LABEL].values)


print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
        predictions = np.add(predictions, rankdata(predict)/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/sample_submission.csv')
submission[LABEL] = predictions
submission.to_csv('rank_averaged_submission.csv', index=False)