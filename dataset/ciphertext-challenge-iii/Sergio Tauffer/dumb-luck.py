# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

### ARE YOU FEELING LUCKY?
# I created this kernel just for fun. It chooses a random ID for each cipher text.
# If you are a beginner, start from here and gradually improve your answer ;)

# Load data
test_data = pd.read_csv('../input/test.csv')
train_data = pd.read_csv('../input/train.csv')

# Create a list with all possible IDs
num_elem = len(test_data)
ids = range(num_elem)

# Pick values randomly
answer = random.sample(ids, num_elem)

# write output file
output = pd.DataFrame({'ciphertext_id':test_data['ciphertext_id'],
                       'index':answer})
output.to_csv('submission.csv', index=False)