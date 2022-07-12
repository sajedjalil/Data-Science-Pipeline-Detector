# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

products = pd.read_csv('../input/products.csv', index_col ='product_id')
orders = pd.read_csv('../input/orders.csv',usecols=['order_id','user_id','eval_set'],index_col='order_id')

item_prior = pd.read_csv('../input/order_products__prior.csv',usecols=['order_id','product_id'],index_col=['order_id','product_id'])

user_product = orders.join(item_prior, how='inner').reset_index().groupby(['user_id','product_id']).count()
