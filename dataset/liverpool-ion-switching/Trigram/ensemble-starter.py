# Credit to Bojan and Rob Mulla #Didn't have time, so made this blend
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
sub1 = pd.read_csv("../input/fe-pipeline-with-histgradientboostingregressor/submission.csv")
sub2 = pd.read_csv("../input/ion-switching-5kfold-lgbm-tracking/sub_script_0.9359188013.csv")
test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
preds_comb = 0.50*sub1.open_channels + 0.50*sub2.open_channels
test['open_channels'] = np.round(np.clip(preds_comb, 0, 10)).astype(int)
test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')