# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import fbeta_score

labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
          'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',
          'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 
          'selective_logging', 'slash_burn', 'water']

def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def find_f2score_threshold(p_valid, y_valid, try_all=False, verbose=False):
    best = 0
    best_score = -1
    totry = np.arange(0,1,0.005) if try_all is False else np.unique(p_valid)
    for t in totry:
        score = f2_score(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True: 
        print('Best score: ', round(best_score, 5), ' @ threshold =', best)
    return best


# Testing ------------------------------------------------------------------

df = pd.read_csv('../input/train.csv')

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

y_true = []
y_pred = []

p = [0.1,0,0,0,0,0.32,0,0,0,0.1,0,0.1,0.28,0,0,0,0.1]

for tags in df.tags[:2000]:
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_true.append(targets)
    p += np.random.uniform(low=0.1, high=0.8, size=(17,))
    p /= np.sum(p)
    y_pred.append(p)

best_threshold = find_f2score_threshold(y_pred, y_true, verbose=True)
