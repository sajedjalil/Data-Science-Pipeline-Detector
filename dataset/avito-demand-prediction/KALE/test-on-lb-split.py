# This script tests on how test data is splited into public and private
# Public  2017-04-12 - 2017-04-13
# Private 2017-04-14 - 2017-04-18

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# fork from https://www.kaggle.com/hakeem/mean-benchmark
df3 = test[['item_id','price','activation_date']].copy()
df3['deal_probability'] =  train['deal_probability'].mean()
# lb score is 0.2663
# mark all probabilities after 2017-04-14 as 0
df3.loc[df3.activation_date >= '2017-04-14', 'deal_probability'] = 0
# score is the same
df3[['item_id','deal_probability']].to_csv('test_on_test.csv', index=False)