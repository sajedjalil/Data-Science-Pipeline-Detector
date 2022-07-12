# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

columns = ['ip','click_time']     
types = {
        'ip':np.int32
        }
            
with open('../input/train.csv','rb') as File:
    data=(pd
            .read_csv(File,
                        parse_dates=['click_time'],
                        infer_datetime_format=True,
                        usecols=columns,
                        dtype=types)
            .reset_index()
            .set_index('click_time')
            .tz_localize('UTC')
            .tz_convert('Asia/Shanghai')
            )
            
end=data.index[-1]
dt=pd.Timedelta('1 days')
dt_sec=dt-pd.Timedelta('1 seconds')
last_day_index=data.loc[end-dt_sec:end,'index']
ante_day_index=data.loc[end-dt-dt_sec:end-dt,'index']
print('the last day of training is from col {} to col {}'.format(last_day_index.min(),last_day_index.max()))
print('the day before last day of training is from col {} to col {}'.format(ante_day_index.min(),ante_day_indexmax()))