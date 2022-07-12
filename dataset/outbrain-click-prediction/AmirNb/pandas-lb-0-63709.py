# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
print (0)
train = pd.read_csv('../input/clicks_train.csv',usecols=['ad_id','clicked'])
print (1)
test = pd.read_csv('../input/clicks_test.csv')
print (2)
ad_likelihood = train.groupby('ad_id')['clicked'].agg(['count','sum']).reset_index()
mean_clicked = train.clicked.mean()
del train
print (3)
w = 0

ad_likelihood['likelihood'] = (ad_likelihood['sum'] + w * mean_clicked) / (ad_likelihood['count'] + w)
print (4)
test = test.merge(ad_likelihood,how='left')
print (5)
test.fillna(mean_clicked,inplace=True)
print (6)
test.sort_values(['display_id','likelihood'],inplace=True,ascending=False)
print (7)
output=test.groupby(['display_id'])['ad_id'].apply(lambda x:' '.join(map(str,x))).reset_index()
print (8)
output.to_csv('output.csv',index=False)
print ('Done!')


