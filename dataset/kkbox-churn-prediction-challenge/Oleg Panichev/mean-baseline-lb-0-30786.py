import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train_v2.csv')
test = pd.read_csv('../input/sample_submission_v2.csv')
# transactions = pd.read_csv('../input/transactions.csv',nrows=1000)
# user_logs = pd.read_csv('../input/user_logs.csv',nrows=1000)
# members = pd.read_csv('../input/members.csv')

print(train.shape)
print(test.shape)
# print(transactions.shape)
# print(user_logs.shape)
# print(members.shape)

m = train['is_churn'].mean()
test['is_churn'] = [m]*test.shape[0]

test.to_csv('subm.csv', index=False)

