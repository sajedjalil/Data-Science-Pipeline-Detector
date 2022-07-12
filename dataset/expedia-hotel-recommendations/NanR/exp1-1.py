# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pandas # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib
#matplotlib.style.use('ggplot')

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

search_base=pandas.read_csv("../input/train.csv",nrows=1000000)
#print (search_base.head(n=5))

sns.countplot(y='site_name',data=search_base)