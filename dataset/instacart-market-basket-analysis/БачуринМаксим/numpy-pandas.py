import numpy as np
import pandas as pd

def purchases_to_str(x):
    if len(x) > 0:
        x = np.array(x, dtype='int').ravel()
        (values, counts) = np.unique(x, return_counts=True)
        ind = np.argsort(counts)[::-1]
        
        n = 8
        if len(x) >= n:
            x = values[ind[:n]]
        
        x_str = ''
        for val in x:
            if val >= 0:
                x_str += str(val) + ' '
            else:
                x_str += 'None '
        x_str = x_str.strip()
    else:
        x_str = 'None'
    return x_str

data_path = '../input/'
order_products_prior = pd.read_csv(data_path + 'order_products__prior.csv')
order_products_train = pd.read_csv(data_path + 'order_products__train.csv')
orders = pd.read_csv(data_path + 'orders.csv')
y = pd.read_csv(data_path + 'sample_submission.csv')

order_products_all = pd.concat([order_products_train, order_products_prior], axis=0)
del order_products_train, order_products_prior

all_data = order_products_all.merge(orders, on='order_id', how='outer')
del order_products_all

user_purchases = all_data.groupby('user_id').product_id.apply(lambda x: purchases_to_str(x))

y = y.merge(all_data, on='order_id', how='left')[['order_id', 'user_id']]
y = y.join(user_purchases, on='user_id')[['order_id', 'product_id']]
y = y.rename(columns={'product_id': 'products'})
y.to_csv('table.csv', index=False)