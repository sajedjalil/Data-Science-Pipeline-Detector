#import os
#os.system("ls ../input")
#os.system("echo \n\n")
#os.system("head ../input/*")

import pandas as pd

df = pd.read_csv('../input/coupon_detail_train.csv', encoding='utf-8')
df.head()