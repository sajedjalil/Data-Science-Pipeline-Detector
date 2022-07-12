N_FOLDS = 5

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
df = pd.read_csv('../input/stage_1_train_labels.csv')
filenames = df.patientId.unique()

fold_ids = np.zeros(len(filenames),dtype=int) + 99
folds = KFold(n_splits= N_FOLDS, shuffle=True, random_state=0).split(filenames)
for i, (train_idx, valid_idx) in enumerate(folds):
    fold_ids[valid_idx] = i
folds_df = pd.DataFrame({'patientId':filenames, 'fold':fold_ids})
df = df.merge(folds_df, on='patientId', how='left')

fold_ids = np.zeros(len(filenames),dtype=int) + 99
folds = KFold(n_splits= N_FOLDS, shuffle=True, random_state=1).split(filenames)
for i, (train_idx, valid_idx) in enumerate(folds):
    fold_ids[valid_idx] = i
folds_df = pd.DataFrame({'patientId':filenames, 'fold2':fold_ids})
df = df.merge(folds_df, on='patientId', how='left')

print(df.head(20))

df.to_csv('stage_1_train_labels_folds.csv')
