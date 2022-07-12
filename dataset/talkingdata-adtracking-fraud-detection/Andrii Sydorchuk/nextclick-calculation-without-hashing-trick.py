# Version 2: Simplify conversion to seconds, as suggested by Md Asraful Kabir.

import time
import numpy as np
import pandas as pd

dtype = {
    'ip': np.int32,
    'app': np.int16,
    'device': np.int16,
    'channel': np.int16,
    'os': np.int16,
    'click_time': object,
}
df = pd.read_csv('../input/train.csv', dtype=dtype, usecols=dtype.keys(), parse_dates=['click_time'], low_memory=True)

start = time.time()
df['nextClick'] = (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - df.click_time).dt.seconds.astype(np.float32)
print('Elapsed: {} seconds'.format(time.time() - start))
