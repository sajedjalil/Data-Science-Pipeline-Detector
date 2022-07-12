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

"""
version of xbgoost installed on kaggle
"""

from sys import version_info
print("Python version:", version_info)

import numpy
print("numpy version", numpy.__version__)

import scipy
print("scipy version", scipy.__version__)

import sklearn
print("sklearn version", sklearn.__version__)

import xgboost
print("XGBoost version", xgboost.__version__)

from multiprocessing import cpu_count
print("CPU count:", cpu_count())

from platform import uname
print('uname:', uname())

with open('/proc/cpuinfo', 'r') as cpuinfo:
    print(cpuinfo.read())