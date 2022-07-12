# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# Any results you write to the current directory are saved as output.

order_products_train_df = pd.read_csv("../input/order_products__train.csv")
order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")
orders_df = pd.read_csv("../input/orders.csv")
products_df = pd.read_csv("../input/products.csv")
aisles_df = pd.read_csv("../input/aisles.csv")
departments_df = pd.read_csv("../input/departments.csv")

def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)

Userid_test = orders_df[orders_df.eval_set == "test"].reindex(columns=['user_id','order_id']).reset_index()
useriddata = pd.merge(Userid_test,orders_df,on ='user_id',how = 'left')
testorders = useriddata[useriddata.eval_set == "prior"].reindex(columns = ['user_id','order_id_y'])
testorders = testorders.rename(columns={'order_id_y': 'order_id'})
reordered_df = pd.merge(testorders,order_products_prior_df , on = 'order_id')
reordered_df = reordered_df[reordered_df.reordered == 1].reindex(columns = ['user_id','order_id','product_id']).groupby('user_id')['product_id'].apply(list)
reordered_df = reordered_df.to_frame().reset_index()
reordered_df = reordered_df.rename(columns= {0: 'user_id'})
data_df = pd.merge(orders_df,reordered_df,on='user_id',how = 'left')
data_df = data_df[data_df.eval_set =="test"].reindex(columns = ['order_id','product_id'])
data_df.to_csv('predicted_with_pandas.csv',header=True, index_label='id')






