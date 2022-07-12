# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

X = pd.read_csv('../input/train.csv')
# print( X.describe() )
placeIdCount = X['place_id'].value_counts() 
print( placeIdCount.size )
print( placeIdCount.head() )
print( placeIdCount.tail() )

plt.figure()
for l in range(0,4):
    XTopPlace = X[ X.place_id == placeIdCount.index[l+100] ]
# XTopPlace.plot(kind='scatter',x = 'x', y = 'accuracy')
    szBin = 1440 * 30 * 6
    nLabels = 1440 * 30 * 12 / szBin
    labelVec =  np.mod( XTopPlace['time'], 1440 * 30 * 12 ) / (szBin)
    
    labelVec = labelVec.round()
    print( labelVec.head() )
    print( labelVec.tail() )
    plt.subplot(2, 2, l+1)
    plt.scatter( XTopPlace['x']  , XTopPlace['y'], c = labelVec ) 
#plt.show()
   
plt.savefig('foo.png')