# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pylab as plt

#for i in range(1, 47+1):
for i in range(1, 10):
    t = []

    if i in [3, 7, 17, 34, 44]:
        mj = 119 
    else:
        mj = 120 

    for j in range(1, mj+1):
        fn = "../input/train/{}_{}_mask.tif".format(i, j)
        im = np.array(plt.imread(fn))

        t.append(im)

    t = np.mean(np.dstack(t), axis=2)
    
    print("patient: {}".format(i))
    
    plt.figure()
    plt.imshow(t)
    plt.title("patient: {}".format(i))
    #plt.show()
    plt.savefig("patient_{}.png".format(i))
