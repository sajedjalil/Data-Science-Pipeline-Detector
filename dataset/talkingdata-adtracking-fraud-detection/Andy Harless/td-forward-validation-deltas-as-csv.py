import numpy as np 
import pandas as pd 
import os

print(os.listdir("../input"))

df = pd.read_pickle('../input/training-and-validation-forward-time-deltas/validation_forward_deltas.pkl.gz')
df.head()

df[['forward_time_delta']].to_csv('td_forward_validation_deltas.csv', index=False)