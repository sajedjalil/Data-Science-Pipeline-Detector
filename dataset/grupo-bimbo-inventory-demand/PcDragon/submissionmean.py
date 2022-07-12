import pandas as pd
import numpy as np

types = {'Demanda_uni_equil': np.uint32}

train_url = "../input/train.csv"
train = pd.read_csv(train_url, usecols=types.keys(), dtype=types)

mean = int(train['Demanda_uni_equil'].mean())
prediction = []
i =0
while (i <= 6999250):
    prediction.append(mean)
    i = i + 1
raws = {'Demanda_uni_equil': prediction}
df = pd.DataFrame(raws, columns=['Demanda_uni_equil'])
df.to_csv("submissionMedian.csv", index_label ='id')