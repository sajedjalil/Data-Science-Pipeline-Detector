import pandas as pd
import numpy as np

data_path = '../input/'

orders = pd.read_csv(data_path + '/orders.csv', usecols=['order_id', 'user_id', 'eval_set', 'order_number']).set_index(
    'order_id')


user_ids_test = orders.loc[orders.eval_set == 'test', 'user_id'].values
print("num of test users: {}".format(len(user_ids_test)))

print("orders: {}".format(orders.shape))
order_products_prior = pd.read_csv(data_path + '/order_products__prior.csv',
                                   usecols=['order_id', 'product_id']).set_index('order_id')
print("order_products_prior: {}".format(order_products_prior.shape))

print("restrict orders to prior orders of test users")
orders_expected = orders[orders.user_id.isin(user_ids_test)]
orders_prior = orders_expected.query("eval_set == 'prior'")
orders_test_expected = orders_expected.query("eval_set == 'test'")


def keep_prior_orders(orders_prior, num_prior_orders):
    orders_prior['num_orders'] = orders_prior.groupby('user_id')['order_number'].transform(max)
    last_orders_prior = orders_prior[(orders_prior['num_orders'] - orders_prior['order_number']) < num_prior_orders]
    return last_orders_prior


print("Use last 10 orders per user")
orders_prior = keep_prior_orders(orders_prior, 11)

print("get prior order products")
order_products_prior = order_products_prior.join(orders_prior['user_id'], how='inner')
print("test user prior order products: {}".format(order_products_prior.shape[0]))


def frequent_items_per_user(order_products_prior, min_support):
    user_product = order_products_prior.groupby(['user_id', 'product_id']).size().to_frame('product_count')

    user_orders_count = orders_prior.groupby('user_id').size().to_frame('order_count')
    user_product = user_product.join(user_orders_count)

    user_product['product_basket_percentage'] = user_product['product_count'] / user_product['order_count']

    frequent_items = (user_product[user_product['product_basket_percentage'] >= min_support]
                      .reset_index(level='product_id')
                      .groupby(level='user_id')['product_id']
                      .apply(list)
                      .to_frame('products')
                      )
    return frequent_items, user_product


print("get products that are in at least 50% of last 10 orders")
min_support = 0.6
frequent_items_per_user, user_product_counts = frequent_items_per_user(order_products_prior, min_support)

print("create expected_products csv")
expected_products = orders_test_expected['user_id'].to_frame().join(frequent_items_per_user, on='user_id')
for row in expected_products.loc[expected_products['products'].isnull(), 'products'].index:
    expected_products.at[row, 'products'] = []
del expected_products['user_id']

expected_products['products'] = expected_products['products'].apply(lambda products: ' '.join(str(p) for p in products))
expected_products.to_csv('expected_products.csv')