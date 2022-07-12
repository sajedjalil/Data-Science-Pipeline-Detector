# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# Any results you write to the current directory are saved as output.
#dtypes = {'platform': np.int8}
cat = pd.read_csv('../input/events.csv')
cat['geo_location'] = cat['geo_location'].str[:2]
#print (len(cat.geo_location.unique()))
print(cat['geo_location'].unique())

#print (cat[cat['platform'] == '\\N'])