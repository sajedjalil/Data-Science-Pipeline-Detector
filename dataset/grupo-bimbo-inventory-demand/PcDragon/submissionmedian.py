import pandas as pd
import numpy as np

types = {'Demanda_uni_equil': np.uint32}

train_url = "../input/train.csv"
train = pd.read_csv(train_url, usecols=types.keys(), dtype=types)

median = int(train['Demanda_uni_equil'].median())
prediction = []
i =0
while (i <= 6999250):
    prediction.append(median)
    i = i + 1
raws = {'Demanda_uni_equil': prediction}
df = pd.DataFrame(raws, columns=['Demanda_uni_equil'])
df.to_csv("submissionMedian.csv", index_label ='id')

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.