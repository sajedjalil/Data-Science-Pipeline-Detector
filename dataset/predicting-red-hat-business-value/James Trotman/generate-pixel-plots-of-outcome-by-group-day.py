# Generate plots of:
# 411 columns: one per train/test day
# N rows: one per group (group_1 from people.csv)
# red pixels mean outcome for that group/day was 1, blue means 0

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.misc import imsave

def generatePixelPlot(df, name):
	print('creating', name)
	rows = []
	cols = []
	data = []
	groupIndex = -1
	prev = -1
	gb = df.groupby(['group_1', 'day_index'])
	for key, df in gb:
		if key[0]!=prev:
			prev = key[0]
			groupIndex += 1
		rows.append(groupIndex)
		cols.append(int(key[1]))
		# simple form of the leak: for a given group/day combination, take the maximum label
		# outcome will be -1, 0, or 1, shift that to 1, 2, 3
		data.append(df.outcome.max()+2)
	m = csr_matrix((data, (rows, cols)), dtype=np.int8)
	codes = m.toarray()
	full = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.int8)
	full[...,0] = codes==3  # red channel is outcome 1
	full[...,2] = codes==2  # blue channel is outcome 0
	'''
	# alternative code to show test set group/day combination as white pixels
	full[...,0] = np.logical_or(codes==1, codes==3)
	full[...,1] = (codes==1)
	full[...,2] = np.logical_or(codes==1, codes==2)
	'''
	imsave(name, full)



people = pd.read_csv("../input/people.csv", usecols=['people_id','group_1'])
people['group_1'] = people['group_1'].apply(lambda s: int(s[s.find(' ')+1:])).astype(int)

train = pd.read_csv("../input/act_train.csv", usecols=['people_id','date','outcome'], parse_dates=['date'])
test = pd.read_csv("../input/act_test.csv", usecols=['people_id','date'], parse_dates=['date'])
test['outcome'] = -1
all = train.append(test)

# make index of days 0..410
epoch = all.date.min()
all['day_index'] = (all['date'] - epoch) / np.timedelta64(1, 'D')

# add group_1 to main DataFrame & sort by it
all = pd.merge(all, people, on='people_id', how='left')
all = all.sort_values('group_1')

# create pixel plots for all groups, in 411x2000 chunks (2000 groups at a time)
groups = all.group_1.unique()
offset = 0
count = 2000
while offset < len(groups):
	sub = groups[offset:offset + count]
	generatePixelPlot(all.ix[all.group_1.isin(sub)], 'groups_%05d_to_%05d.png'%(sub.min(), sub.max()))
	offset += count

# special case: for all (4253) groups that switch outcome over time
# are there any patterns in the timing of changes?
gb = all.groupby('group_1')
switchers = gb.outcome.apply(lambda x: 0 in x.values and 1 in x.values)
groups = set(switchers.ix[switchers].index)
print('#switchers:', len(groups))
generatePixelPlot(all.ix[all.group_1.isin(groups)], 'switcher_groups.png')
