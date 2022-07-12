# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

a = os.listdir("../input/train_1")

#Data = pd.read_csv("../input/train_1/"+a[0])
f = open("../input/train_1/"+a[0],'rb')
for elem in f.read():
    print(elem)

#pd.read_csv("../input/train_1")