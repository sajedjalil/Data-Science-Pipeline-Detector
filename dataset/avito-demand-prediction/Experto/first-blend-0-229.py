
import pandas as pd
import numpy as np

simple = pd.read_csv('../input/avito-lightgbm-starter/simple_mean_benchmark.csv')
tfidf = pd.read_csv("../input/basic-tfidf-on-text-features-0-233-lb/first_attempt.csv")
beep = pd.read_csv("../input/beep-beep/submission.csv")
lgb= pd.read_csv('../input/lightgbm-with-mean-encode-feature-0-233/lgb_with_mean_encode.csv')

b1 = beep.copy()
col = beep.columns

col = col.tolist()
col.remove('item_id')
for i in col:
    b1[i] = (2 * simple[i]  + 2 * tfidf[i] + beep[i] * 4 + lgb[i] * 2  ) /  10
    
b1.to_csv('blend_it.csv', index = False)