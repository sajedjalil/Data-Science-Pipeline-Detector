
import pandas as pd

train = pd.read_csv('../input/train.csv')
print(train.describe())

fe_place_id        = list(set(train["place_id"]))
print("Number of Place ID : " + str(len(fe_place_id)))
