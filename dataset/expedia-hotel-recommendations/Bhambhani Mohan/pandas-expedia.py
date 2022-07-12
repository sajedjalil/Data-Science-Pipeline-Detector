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
# This notebook shows a "most popular local hotel" benchmark implemented with pandas.
# 
# ### Read the train data
# 
# Read in the train data using only the necessary columns. 
# Specifying dtypes helps reduce memory requirements. 
# 
# The file is read in chunks of 1 million rows each. In each chunk we count the number of rows and number of bookings for every destination-hotel cluster combination.
train = pd.read_csv('../input/train.csv',
                    dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    usecols=['srch_destination_id','is_booking','hotel_cluster'],
                    chunksize=1000000)
aggs = []
print('-'*38)
for chunk in train:
    agg = chunk.groupby(['srch_destination_id',
                         'hotel_cluster'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)
    print('.',end='')
print('')
aggs = pd.concat(aggs, axis=0)
aggs.head()
# Next we aggregate again to compute the total number of bookings over all chunks. 
# 
# Compute the number of clicks by subtracting the number of bookings from total row counts.
# 
# Compute the 'relevance' of a hotel cluster with a weighted sum of bookings and clicks.
CLICK_WEIGHT = 0.05
agg = aggs.groupby(['srch_destination_id','hotel_cluster']).sum().reset_index()
agg['count'] -= agg['sum']
agg = agg.rename(columns={'sum':'bookings','count':'clicks'})
agg['relevance'] = agg['bookings'] + CLICK_WEIGHT * agg['clicks']
agg.head()
# ### Find most popular hotel clusters by destination
# 
# Define a function to get most popular hotels for a destination group.
# 
# Previous version used nlargest() Series method to get indices of largest elements. 
# But as @benjamin points out [in his fork](https://www.kaggle.com/benjaminabel/expedia-hotel-recommendations/pandas-version-of-most-popular-hotels/comments) the method is rather slow. 
# I have updated this notebook with a version that runs faster.
def most_popular(group, n_max=5):
    relevance = group['relevance'].values
    hotel_cluster = group['hotel_cluster'].values
    most_popular = hotel_cluster[np.argsort(relevance)[::-1]][:n_max]
    return np.array_str(most_popular)[1:-1] # remove square brackets
# Get most popular hotel clusters for all destinations.
most_pop = agg.groupby(['srch_destination_id']).apply(most_popular)
most_pop = pd.DataFrame(most_pop).rename(columns={0:'hotel_cluster'})
most_pop.head()
# ### Predict for test data
# Read in the test data and merge most popular hotel clusters.
test = pd.read_csv('../input/test.csv',
                    dtype={'srch_destination_id':np.int32},
                    usecols=['srch_destination_id'],)
test = test.merge(most_pop, how='left',left_on='srch_destination_id',right_index=True)
test.head()
# Check hotel_cluster column in test for null values.
test.hotel_cluster.isnull().sum()
# Looks like there's about 14k new destinations in test. Let's fill nas with hotel clusters that are most popular overall.
most_pop_all = agg.groupby('hotel_cluster')['relevance'].sum().nlargest(5).index
most_pop_all = np.array_str(most_pop_all)[1:-1]
most_pop_all
test.hotel_cluster.fillna(most_pop_all,inplace=True)
# Save the submission.
test.hotel_cluster.to_csv('predicted_with_pandas.csv',header=True, index_label='id')
