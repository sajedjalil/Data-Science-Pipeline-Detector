# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

print("# Generate Train and Test")
datadir = '../input'

train_total = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                    dtype={'device_id': np.str})
train_total.drop(["age", "gender"], axis=1, inplace=True)

train_A = train_total[:37323]
train_B = train_total[37323:]

train_D = pd.DataFrame(data=train_B.get_values(), columns=['device_id', 'group'])

print(train_B)
print(train_D)

# for test, will delete later
test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                   dtype={'device_id': np.str})
test["group"] = np.nan

print(test)