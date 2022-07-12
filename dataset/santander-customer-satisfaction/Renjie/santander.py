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
# Read in sample submission
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.head()
# Read in test data
test_data_frame = pd.read_csv("../input/test.csv")
test_data_frame.head()
# Read in train data
raw_data_frame = pd.read_csv("../input/train.csv")
raw_data_frame.head()
raw_data_frame.dtypes
