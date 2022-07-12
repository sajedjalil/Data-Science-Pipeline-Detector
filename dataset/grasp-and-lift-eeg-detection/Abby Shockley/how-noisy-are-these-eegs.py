"""
EEGs tend to be prone to noise in the data. Can we check visually how noisy the data is? This plot is for 
subj1_series8_data. The events start halfway between 2000 and 2200 and end right before 2800. Several of the channels have
a large amount of random noise - P7 for example has regularly occuring spikes that probably don't represent real data. A 
particular challenge for these data sets will be figuring out what's noise and what's real signal.
"""

import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

df_data_subj1_series8 = pd.read_csv('../input/train/subj1_series8_data.csv',header=0)

df_data_subj1_series8.loc[1500:3000,:].plot(subplots=True,figsize=(10,50));

plt.savefig('subj1_series8_data.png')