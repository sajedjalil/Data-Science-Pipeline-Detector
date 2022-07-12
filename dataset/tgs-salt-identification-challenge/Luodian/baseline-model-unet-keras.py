# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))

sub = pd.read_csv("../input/baseline-submission/submission.csv")

sub = sub.loc[:,['id','rle_mask']]

print (sub.head())
sub.to_csv("submission.csv")

# Any results you write to the current directory are saved as output.