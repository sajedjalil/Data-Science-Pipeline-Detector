# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


aisles = pd.read_csv('../input/aisles.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
orders = pd.read_csv('../input/orders.csv')

combined = order_products_prior.merge(orders, on='order_id')

combined.head()