import numpy as np
import pandas as pd

# get sorted unique display_ids
d1=np.unique(pd.read_csv('../input/clicks_train.csv',usecols=['display_id']).display_id.values)
d2=np.unique(pd.read_csv('../input/clicks_test.csv',usecols=['display_id']).display_id.values)
# get counts to prove there are no duplicates
(e,c)=np.unique(pd.read_csv('../input/events.csv',usecols=['display_id']).display_id.values,return_counts=True)
print('Count of non-unique display_id in events:', (c>1).sum())

# merge display_ids from clicks together
i=np.searchsorted(d1,d2)
d=np.insert(d1,i,d2)

# both arrays are sorted, so set equality is easily checked
print('Count of non-matching display_id between merged clicks and events:',(d!=e).sum())
