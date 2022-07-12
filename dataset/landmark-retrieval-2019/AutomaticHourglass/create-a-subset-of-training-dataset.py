# Basically this snippet creates a subset of the training images from most occurring images for playground purposes

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Subset selector for train dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import os

# Counting occurences
df = pd.read_csv('../input/google-landmarks-dataset/train.csv')
df = df.replace(to_replace='None', value=np.nan).dropna()
df['landmark_id'] = df['landmark_id'].astype(int)

c = Counter(df['landmark_id'])
index = []
count = []

for i in c.items():
    index += [i[0]]
    count += [i[1]]

df_counts = pd.DataFrame(index = index,data = count,columns=['counts'])
df_counts = df_counts.sort_values('counts',ascending=False)
df_counts.to_csv('train_counts.csv')


# Find most occurring 500 unique images and take 10 of them
most_occurring = 500
take = 10
selected_index = df_counts.iloc[:most_occurring,:].index

dict_counter = {}
df_train = []
for row in tqdm(df.iterrows()):
    _id,url,landmark_id = row[1]
    if landmark_id in selected_index:
        if landmark_id not in dict_counter:
            dict_counter[landmark_id] = 1
            df_train += [row[1]]
        elif dict_counter[landmark_id] < take:
            dict_counter[landmark_id] += 1
            df_train += [row[1]]
    if all(value == take for value in dict_counter.values()):
        break
    
df_train = pd.DataFrame(df_train)
df_train.to_csv('train_subset.csv',index=False)