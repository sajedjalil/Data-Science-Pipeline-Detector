import numpy as np
import pandas as pd

train = pd.read_csv('../input/training_variants')
classes = sorted(np.unique(train.Class.values))

cnt = train.groupby('Class').ID.agg(['count'])/(train.shape[0])

ss = pd.read_csv('../input/submissionFile')
for c in classes:
	ss['class' + str(c)] = [cnt.loc[c].get('count')] * ss.shape[0]

ss.to_csv('submission.csv', index=False)
