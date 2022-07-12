import sklearn
import pandas as pd
import numpy as np
import scipy as sc


#Importing training Data
rainData = pd.read_csv('../input/train.csv')
print(rainData.head())