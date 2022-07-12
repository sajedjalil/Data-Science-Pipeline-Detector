# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

np.random.seed(2017)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/Train/train.csv')
submission = pd.read_csv('../input/sample_submission.csv')

mean_std = {}
mean_std['adult_males'] = 5
mean_std['subadult_males'] = 4
mean_std['adult_females'] = 26
mean_std['juveniles'] = 15
mean_std['pups'] = 11

print(mean_std)
for c in submission.columns:
    if c != 'test_id':
        submission[c] = (mean_std[c] + 0.5 * np.abs(np.random.randn( len(submission[c]) ) ) ).astype(np.uint8)
submission.to_csv('submission.csv', index=False)
submission.head()

# Any results you write to the current directory are saved as output.
# 15 : 4 3 29 14 9
# 20 : 4 3 26 12 7
# 17 : 4 3 28 14 8