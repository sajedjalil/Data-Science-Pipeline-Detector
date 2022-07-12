import pandas as pd

print("../input/dev_train_basic.csv:")
dev_train_basic = pd.read_csv("../input/dev_train_basic.csv")
print(dev_train_basic.head())

print("../input/id_all_ip.csv:")
ipdata = pd.read_csv('../input/id_all_ip.csv')
print(ipdata.head())

trainip = pd.merge(train,ipdata,left_on='device_id',right_on='device_or_cookie_id')
trainip.rename(columns={'(ip,ip_freq_count,idxip_anonymous_c1,idxip_anonymous_c2,idxip_anonymous_c3,idxip_anonymous_c4,idxip_anonymous_c5)':'ipinfo'}, inplace=True)

print("Train join IP:")
print(trainip.head())