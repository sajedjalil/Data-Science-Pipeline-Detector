import numpy as np 
import pandas as pd
from subprocess import check_output


#read data
orders_all = pd.read_csv("../input/orders.csv",index_col=None,header=0)
orders_prod_p=pd.read_csv("../input/order_products__prior.csv",index_col=None,header=0)
#orders_prod_t=pd.read_csv("MarketBasket/order_products__train.csv",index_col=None,header=0)
#products=pd.read_csv("MarketBasket/products.csv",index_col=None,header=0)

orders_prod_p["current_order"]=orders_prod_p["order_id"]
 



#Create 4 columns for the previous 4 orders to the current order n (n-1,n-2,n-3,n-4)
#The column either hold the order id or -1 if no previous order exists
for i in range(1,5):
    orders_all["n_"+str(i)]=-1



#calculate previous 4 orders ID
orders_all.loc[orders_all["order_number"]>1,"n_1"] =orders_all["order_id"].shift()
orders_all.loc[orders_all["order_number"]>2,"n_2"] =orders_all["order_id"].shift(2)
orders_all.loc[orders_all["order_number"]>3,"n_3"] =orders_all["order_id"].shift(3)
orders_all.loc[orders_all["order_number"]>4,"n_4"] =orders_all["order_id"].shift(4)
orders_all[['order_id',"n_1","n_2","n_3","n_4"]].head()



#join orders with products, so we have the product Id and its previous 4 orders Ids
orders_prod_p= orders_prod_p.set_index("order_id")
orders_all= orders_all.set_index("order_id")
products_all=orders_prod_p.join(orders_all,lsuffix='_order', rsuffix='_product')
del orders_prod_p
del orders_all



products_all[['product_id','current_order',"n_1","n_2","n_3","n_4"]].head()



#initiate columns to save if the products exist in one of the last 4 orders or not
#in_n_1 = Exists in the previous order (n - 1) if 1 "exists", if 0 "doesn't exit", if -1 "no order"
products_all["in_n_1"]=0
products_all["in_n_2"]=0
products_all["in_n_3"]=0
products_all["in_n_4"]=0




#Group Products by Order
g= products_all.groupby('current_order')
#Create dataframe of orderID and tuple of products id
x_n_1_p=g['product_id'].apply(tuple)
products_per_order= pd.DataFrame(list(x_n_1_p.items()))




products_per_order.columns= ['current_order', 'products']
products_per_order.head()




for i in range(1,5):

    products_per_order.columns= ['current_order'+str(i), 'n_'+str(i)+'_p']
    #calculate list of products in previous order i.e n_1_p (n-1 products)
    products_all = pd.merge(products_all, products_per_order,left_on='n_'+str(i),right_on='current_order'+str(i), how='left')
    #calcualte if products exists in previous order
    products_all["in_n_"+str(i)]=  [ -1 if r[2] == -1 else( 1 if r[0] in r[1] else 0)
                               for r in zip(products_all["product_id"], products_all['n_'+str(i)+'_p'],products_all['n_'+str(i)])]
    products_all.drop(["n_"+str(i)+"_p","current_order"+str(i)],axis=1, inplace=True)
    print("in_n_"+str(i) + " Calcuated")    


products_all[['product_id','current_order',"in_n_1","in_n_2","in_n_3","in_n_4"]].head()



