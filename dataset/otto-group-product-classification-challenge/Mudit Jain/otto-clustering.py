import pandas as pd
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

#print(train.head())

master_train=pd.DataFrame([])

for col in train.columns[1:94]:
    temp = pd.DataFrame(train.groupby('target').agg({col:sum})).reset_index()
    master_train=pd.concat([master_train,temp],axis=1)

