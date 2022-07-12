# This is hight of blending !! AGAIN ALL CREDITS GOES TO ORIGINAL KERNEL AUTHORS. I, as a newbie, just created a blend.

import pandas as pd
import numpy as np

#glove = pd.read_csv("../input/toxic-glove/glove.csv")
subb = pd.read_csv('../input/fasttext-like-baseline-with-keras-lb-0-053/submission_bn_fasttext.csv')
ave = pd.read_csv('../input/toxicfile/toxicave.csv')
lstm = pd.read_csv('../input/toxicfiles/baselinelstm0069.csv')
#svm = pd.read_csv("../input/toxic-nbsvm/nbsvm.csv")
#logi = pd.read_csv("../input/logistic-regression-with-words-and-char-n-grams/submission.csv")
ble = pd.read_csv("../input/toxic-simple-blending/blend_sub.csv")
ensem = pd.read_csv("../input/lstm-with-bn-nb-svm-lr-on-conv-ai-lb-0-041/submission_ensemble.csv")
s9821 = pd.read_csv("../input/toxicfile/sub9821.csv")


b1 = subb.copy()
col = subb.columns

col = col.tolist()
col.remove('id')

for i in col:
    b1[i] = np.clip(( subb[i] * ave[i] * lstm[i] * ble[i] * ensem[i] * s9821[i] )**(1/6) , 0.05, 0.95)
    

b1.to_csv('submission32.csv', index = False)





































