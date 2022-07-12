# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

columns_size = pd.read_csv('../input/train_numeric.csv', nrows=1).shape[1]
row_size = pd.read_csv('../input/train_numeric.csv', usecols=['Id']).shape[0]

print(columns_size)
print(row_size)