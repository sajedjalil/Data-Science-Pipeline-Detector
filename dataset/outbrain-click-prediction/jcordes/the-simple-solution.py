# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Only consider ad_ids and how often they are clicked...
train = pd.read_csv('../input/clicks_train.csv',usecols=['ad_id','clicked'])
test = pd.read_csv('../input/clicks_test.csv')

# Group the ads and count up how many views and clicks
ad_likelihood = train.groupby('ad_id')['clicked'].agg(['count','sum']).reset_index()

# Calculate the overall average number of clicks
mean_clicked = train.clicked.mean()
print("overall average click-rate:", mean_clicked)
del train

# Calculate the likelihood of an ad click
ad_likelihood['likelihood'] = (ad_likelihood['sum']) / (ad_likelihood['count'] + 1)

# Left join the ad likelihoods into the training set by ad_id
test = test.merge(ad_likelihood,how='left')

# Fill in the blanks with the overall average likelihood
test.fillna(mean_clicked,inplace=True)

# Sort the rows by the likelihood of each ad_id within each display_id
test.sort_values(['display_id','likelihood'],inplace=True,ascending=False)
print(test.head(20))

# Format the data the way the submission requires
output=test.groupby(['display_id'])['ad_id'].apply(lambda x:' '.join(map(str,x))).reset_index()

# That's it!
output.to_csv('simplesolution2.cvs',index=False)

