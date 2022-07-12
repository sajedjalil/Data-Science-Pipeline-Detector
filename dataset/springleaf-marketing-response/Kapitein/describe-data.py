
import pandas as pd
import csv
train = pd.read_csv("../input/train.csv")
listNums = [len(train[i].value_counts()) for i in train.columns]
with open('listNums.csv','w') as out:
    oout = csv.writer(out)
    for n, i in enumerate(listNums):
        oout.writerow([n,i,train[train.columns[n]].dtype])
