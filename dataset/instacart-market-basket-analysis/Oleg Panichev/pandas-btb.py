import gc
import numpy as np
import pandas as pd

def products_to_str(products):
    # print(products)
    if len(products) > 0:
        products = np.array(products, dtype='int').ravel()
        (values, counts) = np.unique(products, return_counts=True)
        ind = np.argsort(counts)[::-1]
        
        n = 9
        if len(products) >= n:
            products = values[ind[:n]]
            # products = products[:n]
        
        products_str = ''
        for val in products:
            if val >= 0:
                products_str += str(val) + ' '
            elif val == -1:
                if 'None' not in products_str:
                    products_str += 'None '
        products_str = products_str.strip()
    else:
        products_str = 'None'
    return products_str
    
# Read data
data_path = '../input/'
order_products_prior = pd.read_csv(data_path + 'order_products__prior.csv')
order_products_train = pd.read_csv(data_path + 'order_products__train.csv')
orders = pd.read_csv(data_path + 'orders.csv')
ss = pd.read_csv(data_path + 'sample_submission.csv')

order_products_all = pd.concat([order_products_train, order_products_prior], axis=0)
del order_products_train, order_products_prior; gc.collect()

all_data = order_products_all.merge(orders, on='order_id', how='outer')
del order_products_all, orders; gc.collect()

all_data = all_data[all_data.reordered != 0]
all_data = all_data.fillna(-1)
# all_data = all_data.sort_values('add_to_cart_order')
# product_pop = (100*all_data.groupby('product_id').product_id.count()/ \
#     float(len(all_data.product_id))).to_frame()
# product_pop = product_pop.rename(columns={'product_id': 'prod_cnt'})
# print(product_pop.head())
# all_data = all_data.merge(product_pop, on='product_id', how='left')
# print(all_data.columns.values)
# all_data = all_data.sort_values('prod_cnt', ascending=False)
# all_data = all_data[(all_data.add_to_cart_order == 1) | \
#     (all_data.add_to_cart_order == 2) | \
#     (all_data.add_to_cart_order == 3)]
user_products = all_data.groupby('user_id').product_id.apply(lambda x: products_to_str(x))

# Prepare submission
ss = ss.merge(all_data, on='order_id', how='left')[['order_id', 'user_id']]
ss = ss.join(user_products, on='user_id')[['order_id', 'product_id']]
ss = ss.rename(columns={'product_id': 'products'})
ss.to_csv('submission.csv', index=False)
    