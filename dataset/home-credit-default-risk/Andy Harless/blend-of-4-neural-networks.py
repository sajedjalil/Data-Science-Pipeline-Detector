import numpy as np
import pandas as pd 
import os

dirs = [d for d in os.listdir("../input") if d != 'home-credit-default-risk']
print( dirs )
for d in dirs:
    print( os.listdir('../input/'+ d) )
fnames = ['submission.csv', 'sub_nn.csv', 'nn_submission.csv', 'submission.csv']
for i, (d, f) in enumerate(zip(dirs, fnames)):
    preds = pd.read_csv( '../input/' + d + '/' + f
                       ).set_index( 'SK_ID_CURR' ).rename( columns={'TARGET':'pred'+str(i)} )
    if i==0:
        df = preds
        n = len(df)
        df=df.rank().divide(n)
    else:
        df = df.join( preds.rank().divide(n) )
        
result = df.mean(axis=1)
result.name = 'TARGET'
        
pd.DataFrame(result).to_csv('deep_blend_sub.csv')