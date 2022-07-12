# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk # text preprocessing & manipulation
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

from functools import partial # to reduce df memory consumption by applying to_numeric

color = sns.color_palette() # adjusting plotting style
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Import all the data
aisles = pd.read_csv('../input/aisles.csv', engine='c')
departments = pd.read_csv('../input/departments.csv', engine='c')
products = pd.read_csv('../input/products.csv', engine='c')

op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c', 
                       dtype={'order_id': np.int32, 
                              'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 
                              'reordered': np.int8})
                              
op_train = pd.read_csv('../input/order_products__train.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})

# orders
orders = pd.read_csv('../input/orders.csv', engine='c', dtype={'order_id': np.int32, 
                                                           'user_id': np.int32, 
                                                           'order_number': np.int32, 
                                                           'order_dow': np.int8, 
                                                           'order_hour_of_day': np.int8, 
                                                           'days_since_prior_order': np.float16})

# test dataset (submission)
test = pd.read_csv('../input/sample_submission.csv', engine='c')


#Add a field to calculate the sum of times an item was reordered
products['rsum']=op_train.groupby('product_id')['reordered'].sum()
#Add a field to calculate the total times the item could have been reordered
products['rtotal']=op_train.groupby('product_id')['reordered'].count()
#Add a field to calculate the probability that the item was reordered
products['prob']=products['rsum']/products['rtotal']


#Merge all of the details into a goods page
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
# to retain '-' and make product names more "standard"
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 



from functools import partial

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
                             'department_id',
                             'prob']].apply(partial(pd.to_numeric, ##Added the 'prob'
                                                             errors='ignore', 
                                                             downcast='integer')),
                how='left',
                on='product_id'
)

print(order_details.shape, op_train.shape)

# delete (redundant now) dataframes
del op_train

print(order_details.head())

#%%time
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

test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]
last_orders = test_history.groupby('user_id')['order_number'].max()

def get_last_orders_reordered():
    t = pd.merge(
            left=pd.merge(
                    left=last_orders.reset_index(),
                    right=test_history[test_history.reordered == 1],
                    how='left',
                    on=['user_id', 'order_number']
                )[['user_id', 'product_id']],
            right=test_orders[['user_id', 'order_id']],
            how='left',
            on='user_id'
        ).fillna(-1).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(int(e)) for e in set(x)]) 
                                                  ).reset_index().replace(to_replace='-1', 
                                                                          value='None')
    t.columns = ['order_id', 'products']
    return t

# save submission
get_last_orders_reordered().to_csv('last_order_reordered_only.csv', 
                         encoding='utf-8', 
                         index=False)





