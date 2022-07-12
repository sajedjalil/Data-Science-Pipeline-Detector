import pandas as pd

root                          = '../input/'
aisles                        = 'aisles.csv'
departments                   = 'departments.csv'
order_products__prior         = 'order_products__prior.csv'
order_products__train         = 'order_products__train.csv'
orders                        = 'orders.csv'
products                      = 'products.csv'
sample_submission             = 'sample_submission.csv'

df_orders                     = pd.read_csv(root + orders)
df_order_products__prior      = pd.read_csv(root + order_products__prior)
df_order_products__train      = pd.read_csv(root + order_products__train)
df_aisles                     = pd.read_csv(root + aisles)
df_departments                = pd.read_csv(root + departments)
df_products                   = pd.read_csv(root + products)
df_sample_sumission           = pd.read_csv(root + sample_submission)

df_order_products__joinned = df_order_products__prior.append(df_order_products__train)
df_orders_joinned = df_orders.join(df_order_products__joinned.set_index('order_id'), on = 'order_id')
df_temp = df_orders_joinned.join(df_products.set_index('product_id'), on = 'product_id')
df_temp = df_temp.join(df_aisles.set_index('aisle_id'), on = 'aisle_id')
df_complete = df_temp.join(df_departments.set_index('department_id'), on = 'department_id')
df_complete_test = df_complete.loc[df_complete['eval_set'] == 'test']
df_complete_train = df_complete.loc[df_complete['eval_set'] == 'train']
df_complete_prior = df_complete.loc[df_complete['eval_set'] == 'prior']


df_complete.to_csv(root + 'complete.csv', index = False, encoding = 'UTF-8')
df_complete_test.to_csv(root + 'complete_test.csv', index = False, encoding = 'UTF-8')
df_complete_train.to_csv(root + 'complete_train.csv', index = False, encoding = 'UTF-8')
df_complete_prior.to_csv(root + 'complete_prior.csv', index = False, encoding = 'UTF-8')


