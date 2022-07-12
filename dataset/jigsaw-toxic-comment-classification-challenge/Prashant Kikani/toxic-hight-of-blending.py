# This is hight of blending !! AGAIN ALL CREDITS GOES TO ORIGINAL KERNEL AUTHORS. I, as a newbie, just created a blend.

import pandas as pd
import numpy as np

glove = pd.read_csv("../input/toxic-glove/glove.csv")
subb = pd.read_csv('../input/fasttext-like-baseline-with-keras-lb-0-053/submission_bn_fasttext.csv')
ave = pd.read_csv('../input/toxicfile/toxicave.csv')
lstm = pd.read_csv('../input/toxicfiles/baselinelstm0069.csv')
svm = pd.read_csv("../input/toxic-nbsvm/nbsvm.csv")
logi = pd.read_csv("../input/logistic-regression-with-words-and-char-n-grams/submission.csv")
ble = pd.read_csv("../input/toxic-simple-blending/blend_sub.csv")
ensem = pd.read_csv("../input/lstm-with-bn-nb-svm-lr-on-conv-ai-lb-0-041/submission_ensemble.csv")
s9821 = pd.read_csv("../input/toxicfile/sub9821.csv")


b1 = svm.copy()
col = svm.columns

col = col.tolist()
col.remove('id')

for i in col:
    b1[i] = (2 * glove[i] + subb[i] + 2 * svm[i] + ave[i] + lstm[i] + logi[i] + 2 * ble[i] + ensem[i] + 2 * s9821[i]) / 13
    
b1.to_csv('hight_of_blending.csv', index = False)

'''
# Technique by the1owl... Thanks 
sub1 = b1[:]
sub2 = ble[:]
cols = [c for c in ble.columns if c not in ['id','comment_text']]
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in cols:
    blend[c] = np.sqrt(blend[c] * blend[c+'_'])
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]
blend.to_csv('hight_of_blending.csv', index = False)
'''



































