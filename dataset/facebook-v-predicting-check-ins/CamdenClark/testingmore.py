from sklearn.linear_model import LogisticRegression
import pandas as pd
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
import numba
