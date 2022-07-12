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

import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data

cat_names_df = pd.read_csv("../input/category_names.csv")
samp_sub_df = pd.read_csv("../input/sample_submission.csv")


print(cat_names_df.head())
print(samp_sub_df.head())

samp_sub_df["category_id"] = np.random.choice(cat_names_df["category_id"].values,len(samp_sub_df))
samp_sub_df.to_csv("rand_submission.csv", index=False)