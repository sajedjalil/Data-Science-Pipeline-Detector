# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import random
import csv
import os



file_size=700
f=open("../input/clicks_train.csv",'r')
o=open("train_select.csv", 'w')

f.seek(0)
random_line=f.readline()
o.write(random_line)

for i in range(0,20):
    offset=random.randrange(file_size)
    f.seek(offset)
    f.readline()
    random_line=f.readline()
    o.write(random_line)


f.close()
o.close()