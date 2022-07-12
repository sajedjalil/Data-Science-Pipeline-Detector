# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output

training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")
testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")
attribute_data = pd.read_csv("../input/attributes.csv")
descriptions = pd.read_csv("../input/product_descriptions.csv")


attr_list = attribute_data['name'].value_counts()
#print(attr_list)
#print(attribute_data)
attr_list = attribute_data['name'].str.lower().value_counts()
#print(attr_list)
#attr_color = attribute_data['name'].str.findall().value_counts()
value_list = ['color']
attr_color = attribute_data['name'].isin(value_list).value_counts()
#attr_color = value_list.isin(attribute_data['name']).value_counts()
#attr_color = attribute_data['name'].str.lower().str.findall('^.color')
#attr_color = re.search("color", attribute_data['name'].str.lower())

print(attr_color)
