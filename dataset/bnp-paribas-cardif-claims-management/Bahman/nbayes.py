# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv", index_col=['ID'])
test = pd.read_csv("../input/test.csv", index_col=['ID'])
sub = pd.read_csv("../input/sample_submission.csv", index_col=['ID'])
print(train.shape)
print(test.shape)
print(sub.shape)

nb = train.target.mean()

print(nb)

sub['PredictedProb'] = nb

sub.to_csv('nb.csv')

# Any results you write to the current directory are saved as output.