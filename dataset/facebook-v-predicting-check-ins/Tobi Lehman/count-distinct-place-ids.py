# Count the distinct place_ids (the target in the training data)
# also, look at the distribution of distinct targets

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# dataframe holding the training data
df = pd.read_csv("../input/train.csv")

# number of distinct places
print(len(np.unique(df['place_id'])))


# histogram to get idea of frequency of checkins per place
import matplotlib.pyplot as plt
place_freqs = df.groupby('place_id').size()
place_freqs.plot.hist()
