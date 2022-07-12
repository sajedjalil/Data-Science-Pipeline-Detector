import pandas as pd
import numpy as np
import pickle

traincsv = "../input/train.csv"
testcsv = "../input/test.csv"

df_train = pd.read_csv(traincsv)
df_test = pd.read_csv(testcsv)

df_train = df_train._get_numeric_data().fillna(0)
df_test = df_test._get_numeric_data().fillna(0)

print(df_train.head())

pickle.dump(df_train, open("train-filtered.df","wb"))
pickle.dump(df_test, open("test-filtered.df","wb"))