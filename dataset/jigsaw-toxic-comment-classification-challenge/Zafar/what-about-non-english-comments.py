import pandas as pd
import numpy as np
import os
import pickle
data_id = pickle.load(open('../input/best-toxic/not_engl_id.p', 'rb'))  

index_ids = [d[0] for d in data_id]

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
original_sub = pd.read_csv('../input/best-toxic/best.csv')
original_sub.loc[original_sub['id'].isin(index_ids), CLASSES] = 0
original_sub.to_csv('best_tweaked.csv', index=None)