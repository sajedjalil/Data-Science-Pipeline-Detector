# %% [code]
import numpy as np
import pandas as pd

 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pickle

# %% [code]
submission = pd.read_pickle('../input/dataset/dataset/submission.pkl')
submission_val = submission.iloc[:30490, :].set_index("id").T
submission_eva = submission.iloc[30490:, :].set_index("id").T
submission.shape

# %% [code]
df_val = pd.DataFrame()
df_eval = pd.DataFrame()


for i in range(1, 32):
    path = '../input/result-of-fbprophet/output/' + str(i) + '.pkl'
    temp =  pd.read_pickle(path)
    temp_val = temp.iloc[:28, ]
    temp_eval = temp.iloc[28:, ]
    df_val = pd.concat([df_val, temp_val], axis = 1)
    df_eval = pd.concat([df_eval, temp_eval], axis = 1)
    
df_eval.index.name = "id"
df_val.index.name = "id"

# %% [code]
df_val.columns = submission_val.columns
df_eval.columns = submission_eva.columns

df_val = df_val.T
df_eval = df_eval.T

df_val.columns = submission_val.index.values
df_eval.columns = submission_eva.index.values

# %% [code]
final_submission = pd.concat([df_val, df_eval])
final_submission.head()

# %% [code]
final_submission = final_submission.where(final_submission>0.1, 0)

# %% [code]
final_submission.to_csv("submission.csv", float_format='%.3g')