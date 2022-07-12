from collections import Counter

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

df_train_data = pd.read_csv('../input/train_v2.csv')
df_test_data = pd.read_csv('../input/sample_submission_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}

y_full = []
for glob_file in glob('*.csv'):
    submission = pd.read_csv(glob_file)
    y = []
    for f, tags in tqdm(submission.values, miniters=1000):
        preds = np.zeros(17)
        for t in tags.split(' '):
            preds[label_map[t]] = 1
        y.append(preds)
    y = np.array(y, np.uint8)
    y_full.append(y)

y_final = np.array(y_full[0])
for f in range(len(y_full[0])):  # For each file
    for tag in range(17):  # For each tag
        preds = []
        for sub in range(len(y_full)):  # For each submission
            preds.append(y_full[sub][f][tag])
        pred = Counter(preds).most_common(1)[0][0]  # Most common tag prediction among submissions
        y_final[f][tag] = pred

y_final = pd.DataFrame(y_final, columns=labels)

preds = []
for i in tqdm(range(y_final.shape[0]), miniters=1000):
    a = y_final.ix[[i]]
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds
df_test_data.to_csv('final_submission.csv', index=False)
