# This Python script calculates the probability of reoredered 
# given that the product has been ordered p(re|O,P)
# this can be used as a prior. Using Bayesian Updating given a users past orders
# one can find the products which a user is most likely to reorder.
# The Prior is called our_products_prior below and is saved as output

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
pd.set_option('display.float_format', lambda x: '%.4f' % x)

#Now let's get and put the data in  pandas dataframe

prior_product_orders = pd.read_csv('../input/order_products__prior.csv', engine='c', 
            dtype={'order_id': np.int32, 'product_id': np.int32, 
            'add_to_cart_order': np.int16, 'reordered': np.int16})

orders = pd.read_csv('../input/orders.csv', engine='c', 
            usecols = ['order_id','user_id','eval_set','order_number','days_since_prior_order'],
            dtype={'order_id': np.int32, 'user_id': np.int32, 
            'order_number': np.int32, 
            'days_since_prior_order': np.float64})

#products = pd.read_csv('data/products.csv')

prior_product_orders.head(5)

# calculate the prior -- p(re|O,P) = probability of reorder given orders and product
# it is called the Prior which is our belief of reordered before we look at test users

our_products_prior = pd.DataFrame(prior_product_orders.groupby('product_id')['reordered'].agg([('number_of_orders','count'),('sum_of_reorders','sum')]))
our_products_prior['prob_reordered'] = (our_products_prior['sum_of_reorders']+1)/(our_products_prior['number_of_orders']+2)

our_products_prior.head(5)

our_products_prior.to_csv('prior_probability_by_product.csv', encoding='utf-8', index=True)

# next step is to gather test users' orders into a dataframe. We will use this
# to calculate the Bayes factor (or likelihood) for each order by product.
# For the user, each cart (collection of ordered and non-ordered products) will produce a 
# Bayes Factor (BF) for each product in each order


