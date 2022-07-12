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

v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(v)
print(v[2:4])

x = pd.read_csv('../input/train_numeric.csv', nrows=10)

for i, c in enumerate(x.columns):
    print(i, c)