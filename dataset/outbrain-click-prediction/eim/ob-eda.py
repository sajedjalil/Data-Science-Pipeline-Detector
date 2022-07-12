# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np



events = pd.read_csv("../input/events.csv", dtype=np.int32, index_col=0, usecols=[0,3])
events.head()