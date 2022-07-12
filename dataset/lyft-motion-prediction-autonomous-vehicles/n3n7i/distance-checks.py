# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, sys
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sys.path.append("../input/pycent/");

import pyCent


samp = np.random.randint(100, size=[4,10]);

targ = np.random.randint(100, size=[4,10]) + 100;

C = pyCent.liveCentIV(targ);

tl =np.sum(np.abs(samp - targ));
ts =np.sum( (samp - targ)**2 );
tc =np.sum( np.abs( (samp - targ)**3 ));

print("linear Dist:", tl)

print("square Dist:", ts)

print("cubic Dist:", tc)

fval0 = C.weighted_err(samp, targ, np.ones(10), 0.);

print(fval0)


fval = C.weighted_err(samp, targ, np.ones(10));

print(fval)

fval2 = C.weighted_err(samp, targ, np.ones(10), 2.);

print(fval2)

print();

print(fval0 / fval, fval / fval2);

print();


f0l = np.sum(np.abs((samp - np.broadcast_to(fval0, samp.T.shape).T) - targ));
f0s = np.sum( ((samp - np.broadcast_to(fval0, samp.T.shape).T) - targ)**2 );
f0c = np.sum(np.abs( ((samp - np.broadcast_to(fval0, samp.T.shape).T) - targ)**3 ));

print("linear Dist: f0:", f0l, ", impr :", tl - f0l, " e/i ->0 :", f0l / (tl - f0l))
print("square Dist: f0:", f0s, ", impr :", ts - f0s, " e/i ->0 :", f0s / (ts - f0s))
print("cubic Dist: f0:", f0c, ", impr :", tc - f0c, " e/i ->0 :", f0c / (tc - f0c))

print();

xsamp = samp;

for xiter in range(10):

    f1l = np.sum(np.abs((xsamp - np.broadcast_to(fval, xsamp.T.shape).T) - targ));
    f1s = np.sum( ((xsamp - np.broadcast_to(fval, xsamp.T.shape).T) - targ)**2 );
    f1c = np.sum(np.abs( ((xsamp - np.broadcast_to(fval, xsamp.T.shape).T) - targ)**3 ));
    
    
    if(xiter == 9):
    
      print("step:", xiter);

      print("linear Dist: f1:", f1l, ", impr :", tl- f1l, " e/i ->0 :", f1l / (tl - f1l))
    
      print("cubic Dist: f1:", f1c, ", impr :", tc -f1c, " e/i ->0 :", f1c / (tc - f1c))
    
    print("square Dist: f1:", f1s, ", impr :", ts -f1s, " e/i ->0 :", f1s / (ts - f1s))

    xsamp = xsamp - np.broadcast_to(fval, xsamp.T.shape).T;
    
    fval = C.weighted_err(xsamp, targ, np.ones(10));
    
    
    
    
xsamp = samp;

for xiter in range(30):

    f2l = np.sum(np.abs((xsamp - np.broadcast_to(fval2, xsamp.T.shape).T) - targ));
    f2s = np.sum( ((xsamp - np.broadcast_to(fval2, xsamp.T.shape).T) - targ)**2 );
    f2c = np.sum(np.abs( ((xsamp - np.broadcast_to(fval2, xsamp.T.shape).T) - targ)**3 ));

    
    if xiter == 29:
        
      print("step:", xiter);

      print("linear Dist: f2:", f2l, ", impr :", tl - f2l, " e/i ->0 :", f2l / (tl - f2l))

      print("square Dist: f2:", f2s, ", impr :", ts - f2s, " e/i ->0 :", f2s / (ts - f2s))

    print("cubic Dist: f2:", f2c, ", impr :", tc - f2c, " e/i ->0 :", f2c / (tc - f2c))
    
    xsamp = xsamp - np.broadcast_to(fval2, xsamp.T.shape).T;

    fval2 = C.weighted_err(xsamp, targ, np.ones(10), 2.);



