__author__ = 'Nick Sarris (ngs5st)'

import pandas as pd
import operator

# reading data
prior_orders = pd.read_csv('../input/order_products__prior.csv')
train_orders = pd.read_csv('../input/order_products__train.csv')
orders = pd.read_csv('../input/orders.csv')

# removing all user_ids not in the test set
test  = orders[orders['eval_set'] == 'test' ]
user_ids = test['user_id'].values
orders = orders[orders['user_id'].isin(user_ids)]

# combine prior rows by user_id, add product_ids to a list
prior_products = pd.DataFrame(prior_orders.groupby(
    'order_id')['product_id'].apply(list))
prior_products.reset_index(level=['order_id'], inplace=True)
prior_products.columns = ['order_id','products_list']

# combine train rows by user_id, add product_ids to a list
train_products = pd.DataFrame(train_orders.groupby(
    'order_id')['product_id'].apply(list))
train_products.reset_index(level=['order_id'], inplace=True)
train_products.columns = ['order_id','products_list']

# seperate orders into prior/train sets
# turns out there are no test user_ids in the training set so train will be empty
prior = orders[orders['eval_set'] == 'prior']
train = orders[orders['eval_set'] == 'train']

# find the number of the last order placed
prior['num_orders'] = prior.groupby(['user_id'])['order_number'].transform(max)
train['num_orders'] = train.groupby(['user_id'])['order_number'].transform(max)

# merge everything into one dataframe
prior = pd.merge(prior, prior_products, on='order_id', how='left')
train = pd.merge(train, train_products, on='order_id', how='left')
comb = pd.concat([prior, train], axis=0).reset_index(drop=True)

test_cols = ['order_id','user_id']
cols = ['order_id','user_id','order_number','num_orders','products_list']

comb = comb[cols]
test = test[test_cols]

# iterate through dataframe, adding data to dictionary
# data added is in the form of a list:
    # list[0] = weight of the data: (1 + current order number / final order number), thus later data is weighted more
    # list[1] = how important the item is to the buyer: (order in the cart / number of items bought), thus items bought first are weighted more

# also used the average amount of items bought every order as a benchmark for how many items to add per user in the final submission

product_dict = {}
for i, row in comb.iterrows():
    if i % 100000 == 0:
        print('Iterated Through {} Rows...'.format(i))

    if row['user_id'] in product_dict:
        index = 1
        list.append(product_dict[row['user_id']]['len_products'], len(row['products_list']))
        for val in row['products_list']:
            if val in product_dict[row['user_id']]:
                product_dict[row['user_id']][val][0] += 1 + int(row['order_number']) / int(row['num_orders'])
                list.append(product_dict[row['user_id']][val][1], index / len(row['products_list']))
            else:
                product_dict[row['user_id']][val] = [1 + int(row['order_number']) / int(row['num_orders']),
                                              [index / len(row['products_list'])]]
            index += 1
    else:
        index = 1
        product_dict[row['user_id']] = {'len_products': [
            len(row['products_list'])]}
        for val in row['products_list']:
            product_dict[row['user_id']][val] = [1 + int(row['order_number']) / int(row['num_orders']),
                                          [index / len(row['products_list'])]]
            index += 1

final_data = {}
for user_id in product_dict:
    final_data[user_id] = {}
    for product_id in product_dict[user_id]:
        if product_id == 'len_products':
            final_data[user_id][product_id] = \
                round(sum(product_dict[user_id][product_id])/
                    len(product_dict[user_id][product_id]))
        else:
            final_data[user_id][product_id] = \
                [product_dict[user_id][product_id][0],1/
                 (sum(product_dict[user_id][product_id][1])/
                len(product_dict[user_id][product_id][1]))]

# iterate through testing dataframe
# every user_id in test corresponds to a dictionary entry
# call the dictionary with every row, products by weight, combine them into a string, and append them to products

products = []
for i, row in test.iterrows():
    if i % 100000 == 0:
        print('Iterated Through {} Rows...'.format(i))

    final_products = []
    len_products = None
    total_products = final_data[row['user_id']].items()
    for product in total_products:
        if product[0] == 'len_products':
            len_products = product[1]
        else:
            list.append(final_products, product)

    output = []
    product_list = sorted(final_products,
        key=operator.itemgetter(1), reverse=True)
    for val in product_list[:len_products]:
        list.append(output, str(val[0]))
    final_output = ' '.join(output)
    list.append(products, final_output)

# create submission
submission = pd.DataFrame()
submission['order_id'] = test['order_id']
submission['products'] = products
submission.to_csv('submission.csv', index=False)