import pandas as pd
from pandas import Series, DataFrame
import json
import numpy as np
from collections import Counter

traindf = pd.read_json("../input/train.json")
testdf = pd.read_json("../input/test.json")

Cuisine_Type = Counter()
traindf['cuisine'].str.split().apply(Cuisine_Type.update)
Cuisine_Type.most_common(1)

testdf['cuisine'] = "italian"
testdf[['id' , 'cuisine' ]].to_csv("submission_naive.csv", index=False, header= True)
Naive = pd.read_csv("submission_naive.csv")
Naive.head()
