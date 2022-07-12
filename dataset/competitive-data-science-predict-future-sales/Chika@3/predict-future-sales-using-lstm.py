# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# Competition and data overview
# in this competition, we are provided with the challenge of predicting total sales for every product and store in the next month

# The goal of this notebook is to develop a LSTM model for Predict future sales.


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# basic packages        
import numpy as np        
import pandas as pd

# keras model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM


# import all of them
sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv") #read csv files
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
items_cat = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")


# drop duplicates 
sales_train = sales_train.drop_duplicates()     

# find categories
item_cat_unique = items.item_category_id.unique()  

# from this matrix we can find which item_ids belong which item_category
itemId_itemCat = np.zeros([len(items.item_id), len(item_cat_unique)])   

# get unique values
sales_block_unique = sales_train.date_block_num.unique()   
a = sales_train.shop_id.unique()
b = sales_train.item_id.unique()


c = len(items.item_id)

total_shopId_itemId = []
total_item_cnt = []

# make itemId_itemCat so we get adjaction matrix to make relation which they belongs to.
for i in range(len(items.item_id)):
    for j in range(len(item_cat_unique)):
        if items.item_category_id[i] == item_cat_unique[j]:
            itemId_itemCat[items.item_id[i]][item_cat_unique[j]] = 1    

itemCat_itemId= itemId_itemCat.T  # this matrix show in which categories which item_Id is included


# Do linear algebra for dot product to make building block of deep learning model(projecting the vector sum of the element regard to position)
for i in sales_block_unique:
    df2 = sales_train[sales_train['date_block_num'] == i].loc[:] # sort the dataframe according to date_block_num
    df2 = df2[['shop_id', 'item_id', 'item_cnt_day']]  # we only need these subsets values
    df3 = df2.groupby(['shop_id', 'item_id']).sum()  # so if shop_id and item_id duplicates then it sum those values 
    
    item_cnt = np.zeros([len(a), c])   
    shopId_itemId = np.zeros([len(a), c])
    for j in range(len(df3.item_cnt_day)):    
        k, l = df3.item_cnt_day.index[j]    
        item_cnt[k][l] = df3.item_cnt_day.values[j] # no of product sold, measure of monthly amount.
    for m in range(len(itemCat_itemId)):
        shopId_itemId += item_cnt*itemCat_itemId[m]  # this operation basically tell the adding the projection vectors which project a vector onto the space(item_cnt) spanned by columns of itemCat_itemId
    total_item_cnt.append(item_cnt)
    total_shopId_itemId.append(shopId_itemId)  # to get all product sold matrix.


x_train = np.array(total_shopId_itemId[0:33])  # training_dataset
y_train = np.array(total_shopId_itemId[33])
x_test = np.array(total_shopId_itemId[1:34])   # In testing_dataset we take 1-34 training set so we can find 34 date_block_num because x_train & x_test dimention should be same 


x_train1 = x_train.transpose()  # because x_train has (33, 60, 22170) and y_train has (60, 22170) if we take same dim then output_LSTM need 22170 layers so we transpose and then train the model.
y_train1 = y_train.transpose()
x_test1 = x_test.transpose()


# training model
model = Sequential() 
model.add(LSTM(60, dropout=0.2, input_shape=(60,33)))
model.add(Dense(60, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train1, y_train1, epochs=50, batch_size=739)  

# predict on test dataset
pred = model.predict(x_test1)


y_test = []
for i in range(len(test)):
    e = test.shop_id[i]  
    f = test.item_id[i]
    d = pred[f][e]  # according to correct test_ID we find correct pred element 
    y_test.append(d) # then prediction is arange according to test_ID 
    

submission = pd.DataFrame({'ID':test.ID, 'item_cnt_month':y_test}) # this is the final submission
print(submission)    

# store result to csv file
submission.to_csv('datascience_project.csv', index=False)        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session