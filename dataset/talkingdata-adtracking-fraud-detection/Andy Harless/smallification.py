import numpy as np 
import pandas as pd 

model = "../input/pranav-s-r-lightgbm-9683-version/sub_lightgbm_R_reduced.csv"
insub = pd.read_csv(model)
print( insub.head() )

rank = insub['is_attributed'].rank(method='dense')
if rank.max() >= 1e8:
    rank /= 1e8
    ff = '.8f'
else:
    rank /= 1e7
    ff = '.7f'
final_sub = pd.DataFrame()
final_sub['click_id'] = insub['click_id']
final_sub['is_attributed'] = rank
pd.options.display.float_format = ('{:,'+ff+'}').format
print( final_sub.head() )

final_sub.to_csv("smaller.csv.gz", index=False, float_format='%'+ff, compression='gzip')