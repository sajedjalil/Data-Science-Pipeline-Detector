import numpy as np 
import pandas as pd 

###  Ensemble of several kernels mainly from  ###

# Paulo Pinto's Log MA and Days of Week Means (LB: 0.529)
# https://www.kaggle.com/paulorzp/log-ma-and-days-of-week-means-lb-0-529

# Bojan's updates + Lingzhi's updates + Ceshine Lee's LGBM Starter
# https://www.kaggle.com/tunguz/lgbm-one-step-ahead
# https://www.kaggle.com/vrtjso/lgbm-one-step-ahead 
# https://www.kaggle.com/ceshine/lgbm-starter?scriptVersionId=1852107 

# Bojan's CatBoost Starter (LB 0.517)
# https://www.kaggle.com/tunguz/catboost-starter-lb-0-517

filelist = ['../input/ensemble/Median-based.csv',
             '../input/ensemble/lgb.csv',
             '../input/catboost-starter-lb-0-517/cat1.csv']

outs = [pd.read_csv(f, index_col=0) for f in filelist]
concat_df = pd.concat(outs, axis=1)
concat_df.columns = ['sub_ma', 'sub_lgbm','sub_cat']

option=4 # several options to test

if (option==1):
    w = (0.22, 0.44, 0.34)
if (option==2):
    w = (0.22, 0.48, 0.30)
if (option==3):
    w = (0.20, 0.42, 0.38)
if (option==4):
    w = (0.26, 0.38, 0.36)

print(sum(w))
    
concat_df["unit_sales"] = w[0]*concat_df['sub_ma'] + \
                          w[1]*concat_df['sub_lgbm'] + \
                          w[2]*concat_df['sub_cat']
                          
                          
concat_df[["unit_sales"]].to_csv("ensemble_ma_lgbm_cat.csv")