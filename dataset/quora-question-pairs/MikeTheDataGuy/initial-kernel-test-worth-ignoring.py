# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from time import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Just a test

reImport = True
smallRun = False

#Import data
if reImport == True:
    print ("Import Training Data")
    #t0 = time()
    trainData = pd.read_csv("../input/train.csv")
    if smallRun == True:
        trainData = trainData.ix[:10000,:]
    #print("done in %0.3fs." % (time() - t0))
else:
    print ("Training Data Already Imported")

# Any results you write to the current directory are saved as output.