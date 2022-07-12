import pandas as pd

import numpy as np


train_data_raw=pd.read_csv('../input/train.csv')

train_data_raw.info()
train_data_raw.time.plot()