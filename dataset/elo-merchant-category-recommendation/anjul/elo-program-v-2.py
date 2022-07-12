# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_ntran = pd.read_csv("../input/new_merchant_transactions.csv").append(pd.read_csv("../input/historical_transactions.csv"))
data_ntran.isnull().sum()
data_ntran.category_2.fillna(method="bfill", inplace=True)
data_ntran.category_3.fillna(method="bfill", inplace=True)

#Index(['authorized_flag', 'card_id', 'city_id', 'category_1', 'installments',
#       'category_3', 'merchant_category_id', 'merchant_id', 'month_lag',
#       'purchase_amount', 'purchase_date', 'category_2', 'state_id',
#       'subsector_id'],
#      dtype='object')

#Process transaction records
data_ntran_0=data_ntran[["card_id","purchase_date"]]
data_ntran_0["purchase_date"]= pd.to_datetime(data_ntran_0["purchase_date"])
data_ntran_0 = data_ntran_0.groupby("card_id").purchase_date.agg({"min","max"})


data_ntran_1=data_ntran.groupby("card_id")[["city_id"
                                           ,"installments"
                                           ,"merchant_category_id"
                                           ,"merchant_id"
                                           ,"month_lag"
                                           ,"purchase_amount"
                                           ,"state_id"
                                           ]].agg({
                                            "city_id"  : "nunique"
                                           ,"installments"  : "mean"
                                           ,"merchant_category_id" : "nunique"
                                           ,"merchant_id" : "count"
                                           ,"month_lag"   : "mean"
                                           ,"purchase_amount"  : "mean"
                                           ,"state_id"  : "nunique"
                                           })
                                           

data_ntran_1.head()

data_ntran_2=data_ntran[["card_id", "category_3"]]

data_ntran_2["freq"]=1
data_ntran_2.set_index("card_id",inplace=True)

data_ntran_2=data_ntran_2.pivot_table(index="card_id",columns="category_3", values="freq", aggfunc="sum")

data_ntran_2.head()


data_ntran_3=data_ntran[["card_id", "category_2"]]

data_ntran_3["freq"]=1
data_ntran_3.set_index("card_id",inplace=True)

data_ntran_3=data_ntran_3.pivot_table(index="card_id",columns="category_2", values="freq", aggfunc="sum")

data_ntran_3.head()

data_ntran_3=data_ntran_3.rename(columns={"1" : "category_21"
                                ,"2" : "category_22"
                                ,"3" : "category_23"
                                ,"4" : "category_24"
                                ,"5" : "category_25"
                                } )

data_ntran_3.fillna(0, inplace=True)

data_ntran_2.fillna(0, inplace=True)

data_ntran_2.head()
data_ntran=data_ntran_1.join(data_ntran_2, how="left").join(data_ntran_3,how="left").join(data_ntran_0,how="left")

data_train=pd.read_csv("../input/train.csv")

data_train.set_index("card_id",inplace=True)

data_train=data_train.join(data_ntran,how="left")

x_test=pd.read_csv("../input/test.csv")
x_test.set_index("card_id",inplace=True)
x_test=x_test.join(data_ntran_1,how="left").join(data_ntran_2, how="left").join(data_ntran_3,how="left").join(data_ntran_0,how="left")
x_test.first_active_month.fillna(method="bfill",inplace=True)

x_test["first_active_month"]=pd.to_datetime(x_test["first_active_month"])


data_train["first_active_month"]=pd.to_datetime(data_train["first_active_month"])

data_train["month_to_active"]=(data_train["max"]-data_train["first_active_month"]).dt.days

x_test["month_to_active"]=(x_test["max"]-x_test["first_active_month"]).dt.days


data_train["days_bet_trans"]=(data_train["max"]-data_train["min"]).dt.days
x_test["days_bet_trans"]=(x_test["max"]-x_test["min"]).dt.days


data_train["trans_per_day"]=data_train["merchant_id"]/data_train["days_bet_trans"]
x_test["trans_per_day"]=x_test["merchant_id"]/x_test["days_bet_trans"]



data_train_final=data_train[["target", "city_id","installments", 'merchant_category_id',"merchant_id","month_lag","purchase_amount",
                            "state_id","A","B","C",1,2,3,4,5,"month_to_active","days_bet_trans","trans_per_day"]]
                            
data_train_final.rename(columns={1: "cat_21", 2:"cat22" ,3:"cat23" ,4:"cat24" ,5:"cat25"  },inplace=True)

x_test=x_test[["city_id","installments", 'merchant_category_id',"merchant_id","month_lag","purchase_amount",
                            "state_id","A","B","C",1,2,3,4,5,"month_to_active","days_bet_trans","trans_per_day"]]
                            
x_test.rename(columns={1:"cat21" , 2:"cat22" ,3:"cat23" ,4:"cat24" ,5:"cat25"  },inplace=True)

# test train data split
y=data_train_final.iloc[:,0]
x=data_train_final.iloc[:,1:]

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train_scaled=sc.fit_transform(x)
x_test_sc=sc.fit_transform(x_test)

x_test.isnull().sum()
#CALL LASSOCV from sklearn.linear_model

from sklearn.linear_model import LassoCV

elo_model=LassoCV(cv=10).fit(x_train_scaled,y)
y_pred=elo_model.predict(x_test_sc)

#x_test=pd.read_csv("../input/test.csv")
#x_test=x_test.join(data_ntran_1,how="left").join(data_ntran_2, how="left").join(data_ntran_3,how="left").join(data_ntran_0,how="left")

x_test.reset_index(inplace=True)

submission = pd.DataFrame({
        "card_id": x_test["card_id"].values,
        "target": y_pred
    })


submission.to_csv("submission.csv", index=False)
