import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap


df_events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
df_events.head()