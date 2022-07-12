INPUT_DIR = '../input/gs-dense-chexnet-predict-stage-2-from-all-models/'

import numpy as np 
import pandas as pd 
import os
from scipy.special import logit, expit

print(os.listdir("../input"))

fns = []
fns.append('test_preds_pth_fold0_for_combined_folds.csv')
fns = fns + ['test_preds_pth_fold'+str(i)+'.csv' for i in range(1,5)]

fps = [INPUT_DIR + '/' + fn for fn in fns]

preds = []
for i in range(5):
    preds.append( pd.read_csv(fps[i]).set_index('patientId') )

for i,f in enumerate(preds):
    if i:
        df = df.join(f.rename(columns={'targetPredProba':'prob'+str(i)}).drop(['targetPred'],axis=1))
    else:
        df = f.rename(columns={'targetPredProba':'prob'+str(i)}).drop(['targetPred'],axis=1)

results = pd.DataFrame(df.apply(logit).mean(axis=1).apply(expit).rename('prob'))
results.to_csv('test_probs.csv')










