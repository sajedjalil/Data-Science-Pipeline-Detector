import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

csv = pd.read_csv('../input/stage1_sample_submission.csv')

ids = list(csv['id'])
cancer = [np.random.random() for _ in range(len(csv.index))]
submission = pd.DataFrame({'id': ids, 'cancer': cancer})
submission.to_csv('ar_submission.csv', index=False)