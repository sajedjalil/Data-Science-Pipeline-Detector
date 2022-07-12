
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import xgboost as xgb

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

subset0=train.iloc[:, :20]
subset0.count()

subset1=train.iloc[:, 20:40]
subset1.count()
subset1=train.iloc[:, 40:]
subset1.count()
