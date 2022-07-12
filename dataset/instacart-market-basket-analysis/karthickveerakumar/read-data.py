# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Function to read data into Kernel
def read_data(file):
    return pd.read_csv("../input/" + file + ".csv")

#Read data into kernel
#aisles = read_data("aisles") # Kind of like the shelf for a product
#departments = read_data("departments") # Details about the departments the product belongs to
#order_products_prior = read_data("order_products__prior") #Orders
#order_products_train = read_data("order_products__train") # Orders
orders = read_data("orders") # Order details like time, previous orders
#products = read_data("products") # Product details
#sample_submission = read_data("sample_submission")

print(orders.head())
print(orders.describe())
