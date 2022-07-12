# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sns
#print(check_output(["ls", "../input"]).decode("utf8"))
#print(check_output(["ls", "../input/train-jpg"]).decode("utf8"))
sample = pd.read_csv('../input/sample_submission.csv')
print(sample.shape)
sample.head()

df = pd.read_csv('../input/train.csv')
df.head()