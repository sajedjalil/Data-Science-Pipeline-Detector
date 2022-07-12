import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import minmax_scale

submission_1 = pd.read_csv("../input/lgb-gru-lr-lstm-nb-svm-average-ensemble/submission.csv")
submission_2 = pd.read_csv("../input/toxic-one-more-b8bce2/one_more_blend.csv")

blend = submission_1.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
blend[col] = 0.5*minmax_scale(submission_1[col].values)+0.5*minmax_scale(submission_2[col].values)

blend.to_csv("superblend_1.csv", index=False)