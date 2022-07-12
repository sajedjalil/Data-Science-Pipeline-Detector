import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df_submit = pd.read_csv('../input/sample_submission.csv')
df_submit.is_related = np.random.randint(0, 2, df_submit.shape[0])
print(df_submit)
df_submit.to_csv('submission.csv', index=False)
