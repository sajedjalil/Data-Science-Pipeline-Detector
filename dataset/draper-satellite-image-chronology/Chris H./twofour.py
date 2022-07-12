# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

sub = pd.read_csv('../input/sample_submission.csv')
sub['day'] = "4 4 3 2 1"
sub.to_csv('repeat_submit.csv', index=False)
