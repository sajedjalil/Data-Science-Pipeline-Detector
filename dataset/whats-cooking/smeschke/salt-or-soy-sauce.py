import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np

# Reading the data
train = pd.read_json('../input/train.json')

print(train.head())
for cuisine in np.unique(train.cuisine):
    total = 0
    with_salt = 0
    with_soy = 0
    for i in range(len(train.cuisine)):
        if train.cuisine[i]==cuisine:
            total += 1
            if train.ingredients[i].count('salt')>0:
                with_salt += 1
            if train.ingredients[i].count('sauce')>0:
                with_soy += 1
    print(cuisine, with_salt/total)
                

