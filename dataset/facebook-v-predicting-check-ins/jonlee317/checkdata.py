# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/train.csv")
#print(train_data.head())
# contains row_id, x, y, accuracy, time, place_id
#   row_id       x       y  accuracy    time    place_id
#0       0  0.7941  9.0809        54  470702  8523065625
#1       1  5.9567  4.7968        13  186555  1757726713
#2       2  8.3078  7.0407        74  322648  1137537235
#3       3  7.3665  2.5165        65  704587  6567393236
#4       4  4.0961  1.1307        31  472130  7440663949

#print(train_data.shape) 
# contains 29 million data samples  (29118021, 6)

#print(train_data[train_data["place_id"]==8523065625].shape)
# (653, 6)  653 different instances were the place_id is 8523065625

print(train_data[train_data["place_id"]==8523065625].head)

