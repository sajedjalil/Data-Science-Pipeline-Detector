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

import csv
X = []
Y = []

with open("../input/train.csv") as f:
    reader = csv.reader(f)
    
    for i, r in enumerate(reader):
        if i > 0:
            X.append(float(r[1]))
            Y.append(float(r[2]))
            
X = np.array(X).astype(float)
Y = np.array(Y).astype(float)

print(X.min(), X.max(), Y.min(), Y.max())


