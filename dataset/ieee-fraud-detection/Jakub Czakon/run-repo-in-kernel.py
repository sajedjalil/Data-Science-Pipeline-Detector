NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ=='
CONFIG_NAME = 'config_kaggle.yml'

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random((10,10)))
df.to_csv('train_features_v1.csv', index=None)
