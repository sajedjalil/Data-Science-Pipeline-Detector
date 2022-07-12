import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pres
import os
print(os.listdir("../input"))
train_0_fare = pd.read_csv("../input/train-all/train_0_fare.csv")
train_1_fare = pd.read_csv("../input/train-all/train_1_fare.csv")
train_2_fare = pd.read_csv("../input/train-all/train_2_fare.csv")
train_3_fare = pd.read_csv("../input/train-all/train_3_fare.csv")
train_4_fare = pd.read_csv("../input/train-all/train_4_fare.csv")
train_5_fare = pd.read_csv("../input/train-all/train_5_fare.csv")
train_6_fare = pd.read_csv("../input/train-all/train_6_fare.csv")
train_7_fare = pd.read_csv("../input/train-all/train_7_fare.csv")
df = [train_0_fare, train_1_fare, train_2_fare, train_3_fare, train_4_fare, train_5_fare, train_6_fare, train_7_fare]
train_all = pd.concat(df)
train_all.to_csv('train_all.csv', index = 0)