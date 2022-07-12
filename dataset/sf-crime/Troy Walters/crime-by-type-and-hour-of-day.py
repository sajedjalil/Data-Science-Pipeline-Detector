import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

train = pd.read_csv('../input/train.csv', parse_dates = ['Dates'])

train["Hour"] = train.Dates.dt.hour

categories = train.Category.unique()

all_cats = [train[train.Category == cat].Hour.value_counts().reindex(range(24)) for cat in categories]

f, axarr = plt.subplots(13,3, figsize=(6,40))
f.subplots_adjust(right=2.2, hspace=0.5)


for x in range(39):
    axarr[math.floor(x/3), x % 3].plot(all_cats[x])
    axarr[math.floor(x/3), x % 3].set_title(categories[x])

plt.savefig('crimes_by_hour.png', orientation='landscape')