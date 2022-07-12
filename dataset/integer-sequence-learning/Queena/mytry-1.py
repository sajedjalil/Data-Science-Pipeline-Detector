# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
INPUT_TRAIN_FILE='../input/train.csv'
INPUT_TEST_FILE='../input/test.csv'

# Any results you write to the current directory are saved as output.
input_data=pd.read_csv(INPUT_TRAIN_FILE)
input_data = input_data.dropna()
print('input data({0})'.format(input_data.shape))
print(input_data.head())
input_data['Sequence'] = input_data['Sequence'].apply(lambda img: np.fromstring(img, sep=',') )
print('input data({0})'.format(input_data.shape))
print(input_data.head())
train_data = input_data['Sequence'].values
train_label = input_data['Id'].values
print('train_data({0})'.format(train_data.shape))
