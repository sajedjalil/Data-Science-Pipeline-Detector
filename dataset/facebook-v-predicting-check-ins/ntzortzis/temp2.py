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
import datetime
from heapq import nlargest
from operator import itemgetter
import os
import time
import math
from collections import defaultdict


def prep_xy(x, y):
    range = 500
    ix = math.floor(range*x/10)
    if ix < 0:
        ix = 0
    if ix >= range:
        ix = range-1

    iy = math.floor(range*y/10)
    if iy < 0:
        iy = 0
    if iy >= range:
        iy = range-1

    return ix, iy



print('Preparing data...')
f = open("../input/train.csv", "r")
f.readline()
total = 0

grid = defaultdict(lambda: defaultdict(int))
grid_sorted = dict()

    # Calc counts
while 1:
    line = f.readline().strip()
    total += 1
    if line == '':
        break

    arr = line.split(",")
    row_id = arr[0]
    x = float(arr[1])
    y = float(arr[2])
    accuracy = arr[3]
    time = arr[4]
    place_id = arr[5]
    ix, iy = prep_xy(x, y)
    grid[(ix, iy)][place_id] += 1
f.close()
f=itemgetter(1)
print(f(grid))
