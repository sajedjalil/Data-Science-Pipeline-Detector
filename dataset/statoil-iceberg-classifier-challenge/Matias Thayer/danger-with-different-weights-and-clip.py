import pandas as pd
import numpy as np

df1 = pd.read_csv('../input/lb-01400/vggbnw_fcn_en.csv')
df2 = pd.read_csv('../input/submarineering-even-better-public-score-until-now/submission54.csv')

df = pd.merge(df1, df2, on='id')
df['is_iceberg'] = (df['is_iceberg_x'].values*0.6 + df['is_iceberg_y'].values*0.4)
df['is_iceberg'] = np.clip(df['is_iceberg'].values, 0.001, 0.999)
df[['id','is_iceberg']].to_csv('stacking_from_staked.csv', index=False)