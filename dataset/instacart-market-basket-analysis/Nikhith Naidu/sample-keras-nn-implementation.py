
# coding: utf-8

# In[1]:

import numpy
import pandas
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import csv
import sklearn

# In[2]:

trainingOrderItems = pandas.read_csv("../input/order_products__train.csv")
priorOrderItems = pandas.read_csv("../input/order_products__prior.csv")
#aisles = pandas.read_csv("./aisles.csv", )
orders = pandas.read_csv("../input/orders.csv")
#departments = pandas.read_csv("./departments.csv")
products = pandas.read_csv("../input/products.csv")


# In[3]:

# Reorder rate for product ids
productGroup = priorOrderItems.groupby("product_id")
productReorders = productGroup['order_id'].aggregate(numpy.size).to_frame()
productReorders['reordered'] = productGroup['reordered'].aggregate(numpy.mean)
del productGroup


# In[4]:

orderGroup = priorOrderItems.groupby("order_id").aggregate(numpy.max)['add_to_cart_order'].to_frame()
priorOrderItems = pandas.merge(priorOrderItems,orderGroup.rename(columns={'add_to_cart_order':'basket_size'}),left_on='order_id', right_index=True)
priorOrderItems.head()


# In[5]:

# Let's now look at products by user
orderItems = pandas.merge(orders,priorOrderItems, on='order_id')
orderItems['user_product_id'] = 1000000*orderItems['user_id']+orderItems['product_id']
userProducts = orderItems[orderItems['eval_set']=='prior']
userProducts = userProducts[['user_id','order_dow','order_hour_of_day','days_since_prior_order',
                           'product_id','add_to_cart_order', 'user_product_id', 'basket_size', 'order_number', 'order_id']]
userProducts['orders'] = userProducts['user_id']
userAvgProducts = userProducts.groupby('user_product_id').aggregate({'user_id':'mean','order_dow':'mean','order_hour_of_day':'mean',
                                                   'days_since_prior_order':'mean', 'orders':'count',
                                                   'product_id':'mean','add_to_cart_order':'mean', 'basket_size':'mean'})
del orderItems


# In[6]:

lastOrderId = userProducts[userProducts.groupby(['user_product_id'])['order_number'].transform(max) == userProducts['order_number']]
lastOrderId = lastOrderId[['user_product_id','order_id']]
lastOrderId.rename(columns={'order_id':'last_order_id'}, inplace=True)
userAvgProducts = pandas.merge(userAvgProducts, lastOrderId, right_on='user_product_id', left_index=True)
userAvgProducts.set_index('user_product_id', inplace=True)
userAvgProducts.head()


# In[7]:

user_distinct_products = userAvgProducts.groupby('user_id').count()['product_id'].to_frame()
user_distinct_products['user_reordered_products'] = userAvgProducts[userAvgProducts['orders']>1].groupby('user_id').count()['product_id']


# In[8]:

user_distinct_products.rename(columns={'product_id':'distinct_products'},inplace=True)


# In[9]:

userAvgProducts = pandas.merge(user_distinct_products,userAvgProducts,left_index=True,right_on='user_id')


# In[10]:

userAvgProducts['user_reorder_rate'] = userAvgProducts['user_reordered_products']/userAvgProducts['distinct_products']


# In[11]:

# Compile the features
features = userAvgProducts
features.columns = ['user_distinct_products','user_reordered_products','user_id','user_product_dow','user_product_hod','user_product_dsp','user_product_orders','product_id','user_product_addCart', 'user_avg_basket_size', 'user_product_last_order_id','user_reorder_rate']


# In[12]:

userOrders = orders[orders['eval_set']=='prior'].groupby('user_id').aggregate(numpy.max)


# In[14]:

userOrders=userOrders[['order_number']]
features = pandas.merge(userOrders,features, right_on='user_id', left_index=True)


# In[15]:

features['order_number'] = features['user_product_orders']/features['order_number']


# In[16]:

features.rename(columns={'order_number': 'user_product_reorder_rate'}, inplace=True)


# In[17]:

features=pandas.merge(features,productReorders, left_on='product_id',right_index=True)
features.rename(columns={'order_id':'product_total_orders','reordered':'product_reorder_rate'},inplace=True)
features['product_reorders'] = features['product_reorder_rate'] * features['product_total_orders']


# In[18]:

features = pandas.merge(features, products, on='product_id')
features.drop('product_name',axis=1,inplace=True)


# In[19]:

user_order_group = orders[orders['eval_set']=='prior'].groupby('user_id').aggregate({'order_id':'count', 'days_since_prior_order':'mean'})
user_order_group.columns = [['user_orders','user_order_dsp']]
features = pandas.merge(features,user_order_group,left_on='user_id', right_index=True)
features['user_total_products'] = features['user_avg_basket_size']*features['user_orders']


# In[20]:

# Ready the inputs into lightGBM
train_orders = orders[orders['eval_set']=='train']
test_orders = orders[orders['eval_set']=='test']


# In[21]:

train_features = pandas.merge(train_orders, features, on='user_id')
test_features = pandas.merge(test_orders, features, on='user_id')


# In[23]:

del train_orders
del test_orders
del features


# In[24]:

train_features['diff_order_hod'] = abs(train_features['user_product_hod']-train_features['order_hour_of_day']).map(lambda x: min(x, 24-x))
train_features['ratio_dsp'] = train_features['user_product_dsp']/train_features['days_since_prior_order']
train_features['diff_dow'] = abs(train_features['user_product_dow']-train_features['order_dow']).map(lambda x: min (x, 7-x))
train_features['user_product_orders_since_last'] = train_features['user_orders'] - train_features['user_product_last_order_id'].map(orders.order_number)
train_features['user_product_hour_vs_last'] = abs(train_features['order_hour_of_day'] - train_features['user_product_last_order_id'].map(orders.order_hour_of_day)                                                  ).map(lambda x: min(x, 24-x))

test_features['diff_order_hod'] = abs(test_features['user_product_hod']-test_features['order_hour_of_day']).map(lambda x: min(x, 24-x))
test_features['ratio_dsp'] = test_features['user_product_dsp']/test_features['days_since_prior_order']
test_features['diff_dow'] = abs(test_features['user_product_dow']-test_features['order_dow']).map(lambda x: min (x, 7-x))
test_features['user_product_orders_since_last'] = test_features['user_orders'] - test_features['user_product_last_order_id'].map(orders.order_number)
test_features['user_product_hour_vs_last'] = abs(test_features['order_hour_of_day'] - test_features['user_product_last_order_id'].map(orders.order_hour_of_day)                                                  ).map(lambda x: min(x, 24-x))

test_features.head()


# In[25]:

train_features.sort_values(['order_id','product_id'], inplace=True)
test_features.sort_values(['order_id','product_id'], inplace=True)


# In[26]:

train_features.drop(['eval_set','order_number', 'order_number'], axis=1, inplace=True)
test_features.drop(['eval_set','order_number', 'order_number'], axis=1, inplace=True)


# In[28]:

train_orders = orders[orders['eval_set']=='train']
trainProducts = pandas.merge(train_orders,trainingOrderItems, on='order_id')


# In[29]:

trainProducts = trainProducts.groupby('user_id')['product_id'].apply(set)


# In[30]:

trainLabels = []
for row in tqdm(train_features.itertuples()):
    trainLabels += [row.product_id in trainProducts[row.user_id]]
print(len(trainLabels))
print(train_features.shape)


# In[32]:

num_feature_list=['diff_dow','ratio_dsp','diff_order_hod', 'user_avg_basket_size', 'user_product_hour_vs_last',
                  'days_since_prior_order','user_product_reorder_rate','user_product_dow','user_reordered_products',
                  'user_product_hod','user_product_dsp','user_product_orders','user_product_addCart',
                  'product_reorder_rate','product_total_orders', 'user_reorder_rate', 'user_distinct_products',
                  'user_product_orders_since_last']


train_features[num_feature_list] = (train_features[num_feature_list]-train_features[num_feature_list].mean())/((train_features[num_feature_list].max()-train_features[num_feature_list].min()))
test_features[num_feature_list] = (test_features[num_feature_list]-test_features[num_feature_list].mean())/((test_features[num_feature_list].max()-test_features[num_feature_list].min()))

cat_feature_list = []
feature_list = num_feature_list+cat_feature_list


# In[71]:

X_train = train_features[feature_list].fillna(0).as_matrix()[:500000]
Y_train = numpy.array(trainLabels).astype('int8')[:500000]


# In[133]:

# fix random seed for reproducibility
numpy.random.seed(7)

# create model
model = Sequential()
model.add(Dense(13, input_dim=len(feature_list), activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
adam = optimizers.Adam()

# Fit the model
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10,verbose=0, callbacks=[TQDMNotebookCallback()])


# In[151]:

X_test = test_features[feature_list].fillna(0).as_matrix()


# In[152]:

test_preds = model.predict(X_test)


# In[160]:

past_order_id = -1
reorderedProducts = []
output = []
i =0 
maxProd = [0,0]
for row in tqdm(test_features.itertuples()):
    if (row.order_id!=past_order_id):
        if (past_order_id==-1):
            pass
        else:
            if (reorderedProducts == []):
                reorderedProducts.append(maxProd[0])
            output.append([past_order_id," ".join(reorderedProducts)])
            reorderedProducts = []
            maxProd = [0,0]
        past_order_id = row.order_id
    if (test_preds[i]>.2):
            reorderedProducts.append(str(row.product_id))
    else:
        if (test_preds[i] > maxProd[1]):
            maxProd = [str(row.product_id),test_preds[i]]
    i+=1
output.append([past_order_id," ".join(reorderedProducts)])


# In[161]:

output[5]


# In[162]:

with open('predictionNN.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(['order_id','products'])
    for row in output:
        wr.writerow(row)

