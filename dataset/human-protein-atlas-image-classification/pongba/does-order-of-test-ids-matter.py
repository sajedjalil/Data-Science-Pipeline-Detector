# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


df = pd.read_csv('../input/inceptionv3-baseline-lb-0-379/submit_InceptionV3.csv')
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('submit_InceptionV3_shuffle.csv', index=False)