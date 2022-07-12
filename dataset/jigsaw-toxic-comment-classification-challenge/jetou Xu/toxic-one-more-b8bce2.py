'''
The core idea behind all blends is "diversity". 
By blending some moderate model results (with some weights), we can create a more "diverse" stable results.
Errors from one model will be covered by others. Same goes for all the models. 
So, we get more stable results after blendings. 
'''
import pandas as pd
import numpy as np

gru = pd.read_csv("../input/who09829/submission.csv")
gruglo = pd.read_csv("../input/pooled-gru-glove-with-preprocessing/submission.csv")
ave = pd.read_csv("../input/toxic-avenger/submission.csv")
s9821 = pd.read_csv("../input/toxicfile/sub9821.csv")
#glove = pd.read_csv('../input/toxic-glove/glove.csv')
#svm = pd.read_csv("../input/toxic-nbsvm/nbsvm.csv")
best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')

b1 = gru.copy()
col = gru.columns

col = col.tolist()
col.remove('id')

for i in col:
    b1[i] = (gru[i] * 3 + 2 * gruglo[i] + 2 * ave[i] + s9821[i] * 2  + best[i] * 4) /  15
    
b1.to_csv('one_more_blend.csv', index = False)