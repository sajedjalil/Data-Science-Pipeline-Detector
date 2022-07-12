# Replicating v14 (aka v13) of Giulia's notebook:
#   https://www.kaggle.com/giuliasavorgnan/pneumonia-segm-filtered-through-class/notebook?scriptVersionId=6322487

THRESH = 0.225

BOX_PREDICTIONS_FILE = '../input/nonmax2v8-final-stage-2/sub_nms.csv'
CLASS_PREDICTIONS_FILE = '../input/gs-dense-chexnet-predict-stage-2-from-all-models/test_preds_pth_fold0_auc91.csv'
OUTFILE = 'filter199.csv'

import numpy as np 
import pandas as pd
import os

print(os.listdir("../input"))

boxes = pd.read_csv(BOX_PREDICTIONS_FILE).set_index('patientId')
probs = pd.read_csv(CLASS_PREDICTIONS_FILE).set_index('patientId')[['targetPredProba']]

joined = boxes.join(probs)
joined.loc[joined.targetPredProba<THRESH,'PredictionString'] = np.nan

joined[['PredictionString']].to_csv(OUTFILE)