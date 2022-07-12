# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train_file = "../input/train.csv"
test_file = "../input/test.csv"

with open(train_file, 'rb') as fp:
    fp.seek(0,2)
    print(fp.tell())

with open(train_file, 'r') as fp:
    print(fp.readline())
    for i in range(10):
        print(fp.readline())
        
with open(test_file, 'r') as fp:
    print(fp.readline())
    for i in range(10):
        print(fp.readline())
    

# Any results you write to the current directory are saved as output.