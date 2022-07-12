import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

# Load dataset 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')
weather = pd.read_csv('../input/weather.csv')

print (weather)