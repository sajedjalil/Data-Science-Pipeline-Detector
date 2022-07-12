# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import glob
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    
#df = pd.read_csv('../input/train_and_test_data_labels_safe.csv', header=0)
#print(df)
files = sorted(glob.glob("../input/train_1/*.mat"), key=natural_key)

print(len(files))

files = sorted(glob.glob("../input/train_2/*.mat"), key=natural_key)

print(len(files))

files = sorted(glob.glob("../input/train_3/*.mat"), key=natural_key)

print(len(files))