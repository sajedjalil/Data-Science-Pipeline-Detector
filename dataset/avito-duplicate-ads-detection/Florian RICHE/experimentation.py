# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
cat = pd.read_csv("../input/Category.csv")
info_test = pd.read_csv("../input/ItemInfo_test.csv")
info_train = pd.read_csv("../input/ItemInfo_train.csv")
pairs_test = pd.read_csv("../input/ItemPairs_test.csv")
pairs_train = pd.read_csv("../input/ItemPairs_train.csv")
location = pd.read_csv("../input/Location.csv")
sub  = pd.read_csv("../input/Random_submission.csv")

# Any results you write to the current directory are saved as output.

print(cat.head(), cat.shape)
print(info_test.head(), info_test.shape)
print(info_train.head(), info_train.shape)
print(pairs_test.head(), pairs_test.shape)
print(pairs_train.head(), pairs_train.shape)
print(location.head(), location.shape)
print(sub.head(), sub.shape)