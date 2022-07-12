import numpy as np 
import pandas as pd 
import os

print(os.listdir("../input"))

df = pd.read_pickle('../input/bidirectional-talkingdata-test-time-deltas/bidirectional_test_time_deltas.pkl.gz')
df.head()

df[['click_id','forward_time_delta']].to_csv('td_test_forward_time_deltas.csv', index=False)