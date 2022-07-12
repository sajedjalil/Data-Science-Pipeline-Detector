import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Set pandas precision
pd.set_option('precision',1)

#Read in input data into dataframes
train = pd.read_csv("../input/train_ver2.csv", nrows = 10)
test = pd.read_csv("../input/test_ver2.csv", nrows = 10)


print(train.head)