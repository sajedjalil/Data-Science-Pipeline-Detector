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
train = pd.read_csv("../input/train_1.csv")
test = pd.read_csv("../input/key_1.csv")

test['Page'] = test.Page.apply(lambda a: a[:-11])

train['Visits'] = train.drop('Page', axis=1).mean(axis=1, skipna=True)

test = test.merge(train[['Page','Visits']], how='left')
test.loc[test.Visits.isnull(), 'Visits'] = 0

test[['Id','Visits']].to_csv('mean_submission.csv', index=False)