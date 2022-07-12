# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

dataTools = False;
dataTools ="""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
        
class wbqf:
    
    def __init__(self, pathdata, bandwide):
        self.path = pathdata;
        self.band = bandwide;
        
    def lin(self, a,b,x):
        return a*(1-x) + b*x;
    
    def quad(self, a,b,c,x):
        return self.lin(self.lin(a,b,x), self.lin(b,c,x),x);
    
    def surfquad(self, a,b,c,x):
        b2 = self.lin(a,c,0.5);
        b3 = b2 + (b-b2)*2;
        #return self.lin(self.lin(a,b3,x), self.lin(b3,c,x),x);    
        return self.quad(a,b3,c,x);    
    
    def sample(self, start, step):
        ##
        p = self.surfquad(self.path[start,:], self.path[start+int(self.band/2),:], self.path[start+self.band,:], (step-start)/self.band);
        return p;

    #?
    
#testdata = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(5,2);

#wfilter = wbqf(testdata, 4);

#print(wfilter.sample(0,2.3))
#""";

if(dataTools):
    f = open("quadr.py", "w")
    f.write(dataTools);
    f.close();


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session