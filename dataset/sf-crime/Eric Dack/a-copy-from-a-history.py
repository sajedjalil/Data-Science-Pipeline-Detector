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

# ------------------------------------
import zipfile
from time import time

#latitude and longitude of map data
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]
    
drive = '../input/'
z = zipfile.ZipFile(drive+'train.csv.zip')
train = pd.read_csv(z.open('train.csv'))

print(type(train))

#get a unique list of categories
cats = list(set(train.Category))
mapdata = np.loadtxt(drive+"sf_map_copyright_openstreetmap_contributors.txt")

#turn strings into dates
dates = []
datesAll = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in train.Dates])