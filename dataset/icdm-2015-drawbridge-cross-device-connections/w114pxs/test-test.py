import pandas as pd
dev_train = pd.read_csv("../input/dev_train_basic.csv")
print(dev_train['country'].describe())
cookie = pd.read_csv("../input/cookie_all_basic.csv")
print(cookie['country'].describe())
cookie_test=cookie.query('drawbridge_handle != "-1"' )
print(cookie_test['country'].describe())
cookie_train=pd.merge(cookie,cookie_test,on=["drawbridge_handle"])
print(cookie_train['country_x'].describe())
