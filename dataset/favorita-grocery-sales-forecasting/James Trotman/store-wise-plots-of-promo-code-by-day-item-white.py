"""
Visualizing the onpromotion field for all the item/day pairs for a store.

One plot is rendered per store.

Rows represent days, from the earliest dates at the top to the latest dates
and the test set at the bottom.

Columns represent items, ordered by item_nbr. Low item_nbr on the left, high
item_nbr on the right.

WARNING: The files are big! About 4100x1700. best to generate them all
locally and use an image viewer.

Colors:
 - white = no data (whole row omitted, because zero sales)
 - red = onpromotion is True
 - green = onpromotion field missing from row
 - blue = onpromotion is False

This illustrates the nature of the missing onpromotion data. It is hard to
see exact details but it gives a zoomed out overview of the kinds of patterns
present.

The test set has a full grid of day/item/store information, with the
onpromotion field available for each.

The training set has the same but rows with zero sales are omitted, so the
onpromotion field is omitted too.

Visualizing the history of a store, we can see:

 - Earliest data at top (2013), the last 16 rows at the bottom are the test
   set (2017).

 - The first 1.25 years of data the onpromotion field is always missing
   (green pixels).

 - New items are being added regularly, they tend to have higher item numbers
   (right side of image).

 - Some items go on promotion for 1+ weeks (vertical red streak).

 - Some items are cyclically on promotion (1 day per week?).

 - Single days with nearly all items on promotion (a horizontal red streak).

 - Blocks of neighbouring items on promotion, for a block of time, e.g. 1 week.

 - Areas of neighbour products that go out of stock? (White surrounded
   by blue/red).
 
 - Occassional missing days (horizontal white line).
 
 - Some stores with 'fat' horizontal red streaks: a week or more of
   promotions on nearly all items.

 - Browsing the files for all stores, some stores have very similar
   promotional patterns.

The white pixels are 'omitted' training set rows. (In the test set area, last
16 rows, it means the product is not in the test set.)

If you believe the promotional effects to be important, the first big
challenge of the competition is to find a way to impute the onpromotion field
for the rows that were omitted. Imputing it as False will drag predictions
for genuine onpromotion==False rows down.

I believe CF has the full promotion data for all their stores (from March
2014 onwards), so it would be interesting to see what kind of performance
could be reached using the full and complete version of the training data...

P.S. This looks better as a notebook, but when I upload a working notebook,
the Kaggle kernel dies when it starts using matplotlib. "The kernel appears
to have died. It will restart automatically." and no other details.
"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imageio import imwrite

# Read all the rows, subset of columns
def read_data():
    dtypes = {'store_nbr': np.uint8, 'item_nbr': np.uint32}
    cols = ['date','store_nbr','item_nbr','onpromotion']
    train = pd.read_csv('../input/train.csv', dtype=dtypes, usecols=cols, parse_dates=['date'])
    test = pd.read_csv('../input/test.csv', dtype=dtypes, usecols=cols, parse_dates=['date'])
    return train.append(test)

df = read_data()
print(df.shape)
print(df.memory_usage().sum())

t0 = df.date.min()
df['day_ind'] = ((df.date-t0)/np.timedelta64(1, 'D')).astype(np.int16)
df['item_ind'] = LabelEncoder().fit_transform(df.item_nbr).astype(np.int16)
df.drop(['date', 'item_nbr'], axis=1, inplace=True)
gc.collect()

print(df.tail())
print(df.memory_usage().sum())

stores = pd.read_csv('../input/stores.csv', index_col='store_nbr')

gb = df.groupby('store_nbr')

# Render one plot per store
#
# white = no data (whole row omitted, because zero sales)
# red = a promo
# green = promo field missing from row
# blue = not a promo
for store_nbr, store_df in gb:
    
    if store_nbr not in {1,20,25,54}: continue  # remove this line to generate all stores
    
    idstr = '_'.join(map(str, stores.loc[store_nbr].values))
    f = 'promo_s%02d_%s.png'%(store_nbr,idstr)
    ys = store_df.day_ind.values
    xs = store_df.item_ind.values
    h = int(ys.max()+1)
    w = int(xs.max()+1)
    rgb = np.zeros((h, w, 3), dtype=np.uint8) + 255
    rgb[ys, xs, 0] = (store_df.onpromotion.values==True) * 255
    rgb[ys, xs, 1] = store_df.onpromotion.isnull() * 127
    rgb[ys, xs, 2] = (store_df.onpromotion.values==False) * 127
    imwrite(f, rgb)
