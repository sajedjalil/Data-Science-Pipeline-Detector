import pandas as pd
import xgboost as xgb


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.describe())