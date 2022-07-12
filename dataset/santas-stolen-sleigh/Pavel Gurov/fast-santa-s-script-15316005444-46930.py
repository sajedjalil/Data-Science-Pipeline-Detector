# The script created by  https://www.kaggle.com/pgurov
# https://www.kaggle.com/c/santas-stolen-sleigh/leaderboard
# Score: 15316005444.46930

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

from sklearn.cluster import MiniBatchKMeans

gifts = pd.read_csv("../input/gifts.csv")
output_file = 'gurov_sub.csv'

# Split into several parts
print('Mini cluster')
ms = MiniBatchKMeans(n_clusters=10, init='k-means++')
ms.fit(gifts.loc[:, ['Longitude', 'Latitude']].as_matrix())
gifts['cl1'] = ms.labels_

# Every area split into strips
# Every strip sort by Longitude
print('Sort')
gifts.loc[:, 'LatKey'] = np.around(gifts.loc[:, 'Latitude'])
gifts = gifts.sort_values(by=['cl1', 'LatKey', 'Longitude'], ascending=[True, True, True])


# Calculate cusum and split it by 1000 Weight
# 951 - special number,
# because if set 1000, some trips will have a weight of more than 1000
print('Cumsum')
gifts.loc[:, 'Wsum'] = gifts.loc[:, 'Weight'].cumsum()
gifts.loc[:, 'TripId'] = (np.trunc(gifts.loc[:, 'Wsum'] / 951) + 1).astype(int)

# ts = DataFrame(gifts.groupby('TripId')['Weight'].sum())
# print(ts[ts.Weight>1000])

print('file: ' + output_file)
gifts[[ 'GiftId', 'TripId' ]].to_csv(output_file, index = False)


