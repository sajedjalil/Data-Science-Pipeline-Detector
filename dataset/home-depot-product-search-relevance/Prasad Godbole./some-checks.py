
# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io

#load files
training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")
testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")
attribute_data = pd.read_csv("../input/attributes.csv")
descriptions = pd.read_csv("../input/product_descriptions.csv")

# merge descriptions
training_data = pd.merge(training_data, descriptions, on="product_uid", how="left")
print(str(training_data.info()))
