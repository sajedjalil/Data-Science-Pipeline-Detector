import pandas as pd
import numpy as np

data_path = '../input'

# read relevant csvs
orders = pd.read_csv(data_path + '/orders.csv', usecols=['order_id', 'user_id', 'eval_set', 'order_number']).set_index('order_id')
print("orders: {}".format(orders.shape))
order_products_prior = pd.read_csv(data_path + '/order_products__prior.csv', usecols=['order_id', 'product_id']).set_index('order_id')
print("order_products_prior: {}".format(order_products_prior.shape))

# test users have a last order with eval_set == test
user_ids_test = orders.loc[orders.eval_set == 'test', 'user_id'].values
print("num of test users: {}".format(len(user_ids_test)))

print("restrict orders to prior orders of test users")
orders_test_user = orders[orders.user_id.isin(user_ids_test)]
orders_prior = orders_test_user.query("eval_set == 'prior'")
orders_test = orders_test_user.query("eval_set == 'test'")

def keep_n_prior_orders(orders_prior, num_prior_orders):
    orders_prior['num_orders'] = orders_prior.groupby('user_id')['order_number'].transform(max)
    latest_orders_prior = orders_prior[(orders_prior['num_orders'] - orders_prior['order_number']) < num_prior_orders]
    return latest_orders_prior

print("Use last 10 orders per user")
orders_prior = keep_n_prior_orders(orders_prior, 10)

#order_products_prior = order_products.loc[orders_prior.index, ['user_id', 'product_id']]
print("get prior order products")
order_products_prior = order_products_prior.join(orders_prior['user_id'], how='inner')
print("test user prior order products: {}".format(order_products_prior.shape[0]))

def frequent_items_per_user(order_products_prior, min_support):
    user_product = order_products_prior.groupby(['user_id', 'product_id']).size().to_frame('product_count')
    #user_product['total_ordered_products'] = user_product.groupby(level='user_id')['product_count'].transform(sum)
    
    # recount number of orders per user
    user_orders_count = orders_prior.groupby('user_id').size().to_frame('order_count')
    user_product = user_product.join(user_orders_count)

    user_product['product_basket_percentage'] = user_product['product_count'] / user_product['order_count']
    
    # get all products with support >= min_support
    frequent_items = (user_product[user_product['product_basket_percentage'] >= min_support]
                      .reset_index(level='product_id')
                      .groupby(level='user_id')['product_id']
                      .apply(list)
                      .to_frame('products')
                     )
    return frequent_items, user_product

print("get products that are in at least 50% of last 10 orders")
min_support=0.5
frequent_items_per_user, user_product_counts = frequent_items_per_user(order_products_prior, min_support)

print("create submission csv")
submission = orders_test['user_id'].to_frame().join(frequent_items_per_user, on='user_id')
# set empty list for users that have no products with support >= min_suppport
for row in submission.loc[submission['products'].isnull(), 'products'].index:
    submission.at[row, 'products'] = []
del submission['user_id']

submission['products'] = submission['products'].apply(lambda products : ' '.join(str(p) for p in products))
submission.to_csv('submission.csv')