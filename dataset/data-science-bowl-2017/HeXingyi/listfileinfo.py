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

import os

#os.listdir("../input")

df_train=pd.read_csv("../input/stage1_labels.csv")
df_train.head()
print("Number of training patients: {}".format(len(df_train)))

print("Number of sample images: {}".format(len(os.listdir("../input/sample_images"))))

df_submission=pd.read_csv("../input/stage1_sample_submission.csv")
df_submission.head()
print("Number of sample submission: {}".format(len(df_submission)))




