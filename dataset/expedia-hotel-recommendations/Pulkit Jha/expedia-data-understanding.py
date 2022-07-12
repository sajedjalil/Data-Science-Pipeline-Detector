# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from os import system
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_test = pd.read_csv("../input/test.csv",header=0,sep=",")
#print(check_output(["head","../input/train.csv"]).decode("utf8"))
#print(df_test.info())
#print(df_test.head())

print(system("head ../input/train.csv"))
print(system("wc -l ../input/train.csv"))