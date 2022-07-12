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
