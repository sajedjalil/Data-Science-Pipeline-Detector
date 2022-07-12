import pandas as pd
import os
import numpy

os.system("ls ../input")

train = pd.read_csv("../input/train.csv", header=0, parse_dates=[0])
test = pd.read_csv("../input/test.csv", header=0, parse_dates=[1])
data = train.append(test)
data["Year"] = data.Date.map(lambda d: d.year)


print(data.pivot_table(index=["Year", "Species"], values=["WnvPresent"], aggfunc=[numpy.mean, len]))