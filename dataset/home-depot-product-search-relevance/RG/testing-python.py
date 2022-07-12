# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gzip
import io

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
sample_submission = pd.read_csv("../input/sample_submission.csv")
print(sample_submission.head(5))
product_description = pd.read_csv("../input/product_descriptions.csv")
print(product_description.head(5))
attributes = pd.read_csv("../input/attributes.csv")
print(attributes.head(5))
train = pd.read_csv("../input/train.csv", sep=",", encoding="ISO-8859-1")
print(train.head(10))
test = pd.read_csv("../input/test.csv", sep=",", encoding="ISO-8859-1")
print(test.head(10))