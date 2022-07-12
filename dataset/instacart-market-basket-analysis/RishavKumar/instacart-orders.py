# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:54:39 2017

@author: rkumar
"""

import pandas as pd # dataframes
import numpy as np # algebra & calculus
import nltk # text preprocessing & manipulation
# from textblob import TextBlob
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

from functools import partial # to reduce df memory consumption by applying to_numeric

color = sns.color_palette() # adjusting plotting style
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings

%matplotlib inline
aisles = pd.read_csv('../input/aisles.csv', engine='c')
departments = pd.read_csv('../input/departments.csv', engine='c')
product= pd.read_csv('../input/products.csv',engine='c')


product.head(5)
aisles.head(5)
departments.head(5)
goods =pd.merge(left=pd.merge(left=product,right=departments,how='left'),right=aisles, how='left')
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 

goods.head(5)
## read the order data set
orders = pd.read_csv('../input/orders.csv', engine='c', 
                     dtype={'order_id': np.int32,'user_id': np.int32,
                     'order_number': np.int32,'order_dow': np.int8, 
                    'order_hour_of_day': np.int8,'days_since_prior_order': np.float16})
print('Total orders: {}'.format(orders.shape[0]))
orders.head()
## read the product prior data set
op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
print('Total ordered products(prior): {}'.format(op_prior.shape[0]))
op_prior.head()

## read the train data set
op_train = pd.read_csv('../input/order_products__train.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
print('Total ordered products(train): {}'.format(op_train.shape[0]))
op_train.head(10)

# merge train and prior together iteratively, to fit into 8GB kernel RAM
# split df indexes into parts
indexes = np.linspace(0, len(op_prior), num=10, dtype=np.int32)

# initialize it with train dataset
order_details = pd.merge(
                left=op_train,
                 right=orders, 
                 how='left', 
                 on='order_id'
        ).apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))

# add order hierarchy
order_details = pd.merge(
                left=order_details,
                right=goods[['product_id', 
                             'aisle_id', 
                             'department_id']].apply(partial(pd.to_numeric, 
                                                             errors='ignore', 
                                                             downcast='integer')),
                how='left',
                on='product_id'
)

print(order_details.shape, op_train.shape)

# delete (redundant now) dataframes
del op_train

order_details.head()

%%time
# update by small portions
for i in range(len(indexes)-1):
    order_details = pd.concat(
        [   
            order_details,
            pd.merge(left=pd.merge(
                            left=op_prior.loc[indexes[i]:indexes[i+1], :],
                            right=goods[['product_id', 
                                         'aisle_id', 
                                         'department_id' ]].apply(partial(pd.to_numeric, 
                                                                          errors='ignore', 
                                                                          downcast='integer')),
                            how='left',
                            on='product_id'
                            ),
                     right=orders, 
                     how='left', 
                     on='order_id'
                ) #.apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))
        ]
    )
        
print('Datafame length: {}'.format(order_details.shape[0]))
print('Memory consumption: {:.2f} Mb'.format(sum(order_details.memory_usage(index=True, 
                                                                         deep=True) / 2**20)))
# check dtypes to see if we use memory effectively
print(order_details.dtypes)

# make sure we didn't forget to retain test dataset :D
test_orders = orders[orders.eval_set == 'test']

# delete (redundant now) dataframes
del op_prior, orders

test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]\
.groupby('user_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()
test_history.columns = ['user_id', 'products']

test_history = pd.merge(left=test_history, 
                        right=test_orders, 
                        how='right', 
                        on='user_id')[['order_id', 'products']]

test_history.to_csv('../dumb_submission.csv', encoding='utf-8', index=False)
# Any results you write to the current directory are saved as output.