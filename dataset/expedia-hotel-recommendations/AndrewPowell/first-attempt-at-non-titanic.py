# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_full = pd.read_csv('../input/train.csv', usecols=['srch_destination_id','is_booking','hotel_cluster'], nrows=1000000)

train = train_full

mode_cluster = train["hotel_cluster"].mode()

print(mode_cluster[0])

test = pd.read_csv('../input/test.csv', usecols=['srch_destination_id'])

print(train.head())

print(len(train["srch_destination_id"].value_counts()))

print(len(train["hotel_cluster"].value_counts()))

dest_ids = train["srch_destination_id"].unique()

print(dest_ids)

dest_cluster = {}
count = 0

for row in train["srch_destination_id"]:
    if (row not in dest_cluster):
        dest_cluster[row] = [train["hotel_cluster"].iloc[count]]
    else:
        dest_cluster[row].append(train["hotel_cluster"].iloc[count])
    count += 1
    
dest_modes = {}

for key in dest_cluster:
    dest_list = Counter(dest_cluster[key])
    temp =  dest_list.most_common(5)
    if (len(temp) == 1):
        dest_modes[key] = temp[0][0]
    elif (len(temp) == 2):
        dest_modes[key] = np.array_str(np.array([temp[0][0],temp[1][0]]))[1:-1]
    elif (len(temp) == 3):
        dest_modes[key] = np.array_str(np.array([temp[0][0],temp[1][0],temp[2][0]]))[1:-1]
    elif (len(temp) == 4):
        dest_modes[key] = np.array_str(np.array([temp[0][0],temp[1][0],temp[2][0],temp[3][0]]))[1:-1]
    else:
        dest_modes[key] = np.array_str(np.array([temp[0][0],temp[1][0],temp[2][0],temp[3][0],temp[4][0]]))[1:-1]
    
predictions = []

for row in test["srch_destination_id"]:
    if (row in dest_modes.keys()):
        predictions.append(dest_modes[row])
    else:
        predictions.append(mode_cluster[0])
        
print(len(predictions))
    
submission = pd.DataFrame({
        "hotel_cluster": predictions
    })
    
submission.to_csv("kaggle.csv", index_label='id')