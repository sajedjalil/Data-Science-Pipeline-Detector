# All credits goes to original authors.. Just another blend...
import pandas as pd
from sklearn.preprocessing import minmax_scale
sup = pd.read_csv('../input/blend-of-blends-1/superblend_1.csv')
allave = pd.read_csv('../input/lgb-gru-lr-lstm-nb-svm-average-ensemble/submission.csv')
gru = pd.read_csv('../input/bi-gru-cnn-poolings/submission.csv')

blend = allave.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.2*minmax_scale(allave[col].values)+0.6*minmax_scale(gru[col].values)+0.2*minmax_scale(sup[col].values)
print('stay tight kaggler')
blend.to_csv("hight_of_blend_v2.csv", index=False)