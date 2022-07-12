import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

submission = pd.read_csv('../input/Random_submission.csv')
submission['probability']=0.0
submission.to_csv('0_baseline.csv', header=True, index_label='id')
# Any results you write to the current directory are saved as output.