import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))

df_test = pd.read_csv("../input/test.csv")
print(df_test)
df_train = pd.read_csv("../input/train.csv")
print(df_train)

df_test.head()
df_train.head()
df_train.target.value_counts()

