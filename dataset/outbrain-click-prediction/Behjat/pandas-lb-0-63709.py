# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

t0 = time.time()
train = pd.read_csv('../input/clicks_train.csv',usecols=['ad_id','clicked'])

print ('1: ' + str(time.time()-t0))
t0 = time.time()


test = pd.read_csv('../input/clicks_test.csv')

print ('2: ' + str(time.time()-t0))
t0 = time.time()

ad_likelihood = train.groupby('ad_id')['clicked'].agg(['count','sum','mean']).reset_index()

print ('3: ' + str(time.time()-t0))
t0 = time.time()

mean_clicked = train.clicked.mean()

print ('4: ' + str(time.time()-t0))
t0 = time.time()

del train
ww = 0
ad_likelihood['likelihood'] = (ad_likelihood['sum'] + ww * mean_clicked) / (ad_likelihood['count'] + ww)

print ('5: ' + str(time.time()-t0))
t0 = time.time()

test = test.merge(ad_likelihood,how='left')

print ('6: ' + str(time.time()-t0))
t0 = time.time()

test.fillna(mean_clicked,inplace=True)

print ('7: ' + str(time.time()-t0))
t0 = time.time()


test.sort_values(['display_id','likelihood'],inplace=True,ascending=False)

print ('8: ' + str(time.time()-t0))
t0 = time.time()

output=test.groupby(['display_id'])['ad_id'].apply(lambda x:' '.join(map(str,x))).reset_index()

print ('9: ' + str(time.time()-t0))
t0 = time.time()

output.to_csv('output.csv',index=False)

print ('10: ' + str(time.time()-t0))
t0 = time.time()