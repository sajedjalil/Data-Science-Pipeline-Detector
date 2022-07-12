# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# load files
training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")
testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")
attribute_data = pd.read_csv("../input/attributes.csv")
descriptions = pd.read_csv("../input/product_descriptions.csv")

# merge descriptions
training_data = pd.merge(training_data, descriptions, on="product_uid", how="left")

# merge product counts
product_counts = pd.DataFrame(pd.Series(training_data.groupby(["product_uid"]).size(), name="product_count"))
training_data = pd.merge(training_data, product_counts, left_on="product_uid", right_index=True, how="left")

# merge brand names
brand_names = attribute_data[attribute_data.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand_name"})
training_data = pd.merge(training_data, brand_names, on="product_uid", how="left")
training_data.brand_name.fillna("Unknown", inplace=True)

print(str(training_data.info()))
print(str(training_data.describe()))
training_data[:50]