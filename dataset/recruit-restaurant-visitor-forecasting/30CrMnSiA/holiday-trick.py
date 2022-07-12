# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


"""
Created on Mon Feb  5 23:27:10 2018

@author: hzs
"""

import pandas as pd #
import numpy as np

first_sub = pd.read_csv("first_sub.csv")

first_sub['tmp'] = np.nan
         
first_sub['air_store_id'] = first_sub.id.map(lambda x: '_'.join(x.split('_')[:-1]))
first_sub['date'] = first_sub.id.map(lambda x: x.split('_')[2])
first_sub['date'] =first_sub['date'].astype('datetime64[ns]')

first_sub.loc[first_sub.date=='2017-5-3','tmp'] = np.sqrt(first_sub.loc[first_sub.date=='2017-4-29','visitors'].values*first_sub.loc[first_sub.date=='2017-5-13','visitors'].values)
first_sub.loc[first_sub.date=='2017-5-4','tmp'] = np.sqrt(first_sub.loc[first_sub.date=='2017-4-29','visitors'].values*first_sub.loc[first_sub.date=='2017-5-13','visitors'].values)
first_sub.loc[first_sub.date=='2017-5-5','tmp'] = np.sqrt(first_sub.loc[first_sub.date=='2017-4-29','visitors'].values*first_sub.loc[first_sub.date=='2017-5-13','visitors'].values)
first_sub.loc[first_sub.date=='2017-5-2','tmp'] = np.sqrt(first_sub.loc[first_sub.date=='2017-4-28','visitors'].values*first_sub.loc[first_sub.date=='2017-5-12','visitors'].values)

#conservative
#==============================================================================
# first_sub.loc[first_sub.date=='2017-5-3','visitors'] = np.sqrt(first_sub.loc[first_sub.date=='2017-5-3','tmp']*first_sub.loc[first_sub.date=='2017-5-3','visitors'])
# first_sub.loc[first_sub.date=='2017-5-4','visitors'] = np.sqrt(first_sub.loc[first_sub.date=='2017-5-4','tmp']*first_sub.loc[first_sub.date=='2017-5-4','visitors'])
# first_sub.loc[first_sub.date=='2017-5-5','visitors'] = np.sqrt(first_sub.loc[first_sub.date=='2017-5-5','tmp']*first_sub.loc[first_sub.date=='2017-5-5','visitors'])
# first_sub.loc[first_sub.date=='2017-5-2','visitors'] = np.sqrt(first_sub.loc[first_sub.date=='2017-5-2','tmp']*first_sub.loc[first_sub.date=='2017-5-2','visitors'])
#==============================================================================

#radical
first_sub.loc[first_sub.date=='2017-5-3','visitors'] = first_sub.loc[first_sub.date=='2017-5-3','tmp']
first_sub.loc[first_sub.date=='2017-5-4','visitors'] = first_sub.loc[first_sub.date=='2017-5-4','tmp']
first_sub.loc[first_sub.date=='2017-5-5','visitors'] = first_sub.loc[first_sub.date=='2017-5-5','tmp']
first_sub.loc[first_sub.date=='2017-5-2','visitors'] = first_sub.loc[first_sub.date=='2017-5-2','tmp']

first_sub[['id','visitors']].to_csv('second_sub.csv', float_format='%.5f', index=None)