'''
The core idea behind all blends is "diversity". 
By blending some moderate model results (with some weights), we can create a more "diverse" stable results.
Errors from one model will be covered by others. Same goes for all the models. 
So, we get more stable results after blendings. 
'''
import pandas as pd
import numpy as np

gru = pd.read_csv("../input/pooled-gru-with-preprocessing/submission.csv")
grucnn = pd.read_csv('../input/bi-gru-cnn-poolings/submission.csv')
gruglo = pd.read_csv("../input/pooled-gru-glove-with-preprocessing/submission.csv")
ave = pd.read_csv("../input/toxic-avenger/submission.csv")
s9821 = pd.read_csv("../input/toxicfile/sub9821.csv")
glove = pd.read_csv('../input/toxic-glove/glove.csv')
svm = pd.read_csv("../input/toxic-nbsvm/nbsvm.csv")
best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')
lstmcnn = pd.read_csv('../input/bidirectional-lstm-with-convolution/submission.csv')
bbest = pd.read_csv('../input/oof-stacking-regime/submission.csv')

b1 = svm.copy()
col = svm.columns

col = col.tolist()
col.remove('id')
print('hi there, fine? enjoying blending huh? yep blends are awesome !!')

# LET'S KEEP IT SUPER SIMPLE. EQUAL WEIGHT TO ALL BASE MODELS. LET SEE WHAT HAPPENS.
# (2 * gru[i]  + 2 * gruglo[i] + grucnn[i] * 4 + ave[i] + s9821[i] * 2 + best[i] * 3 + 4 * lstmcnn[i]) /  18 
# ABOVE WEIGHT COMBINATION WAS GIVING 0.9861 ON LB. 
# (gru[i]  + gruglo[i] + grucnn[i] + ave[i] + s9821[i] + best[i] + lstmcnn[i]) /  7 
# ABOVE SIMPLE AVERAGING GIVES 0.9859

for i in col:
    b1[i] = (2 * gru[i]  + 2 * gruglo[i] + grucnn[i] * 4 + ave[i] + s9821[i] * 2 + best[i] * 3 + 4 * lstmcnn[i] + 6 * bbest[i]) /  24
    
b1.to_csv('one_more_blend.csv', index = False)
'''
sub1 = b1[:]
sub2 = best[:]
col = [c for c in gru.columns if c not in ['id','comment_text']]
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in col:
    blend[c] = np.sqrt(blend[c] * blend[c+'_'])
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]
blend.to_csv('one_more_blend.csv', index=False)
'''
