import numpy as np
import pandas as pd
pd.DataFrame({'Path':np.concatenate(([0], pd.read_csv('../input/cities.csv').iloc[1:].sort_values(['Y','X'])['CityId'].values, [0]))}).to_csv('very_simple_baseline.csv', index=False)