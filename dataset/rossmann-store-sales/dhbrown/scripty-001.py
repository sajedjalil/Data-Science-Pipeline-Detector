#!/usr/bin/env python

import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.head())