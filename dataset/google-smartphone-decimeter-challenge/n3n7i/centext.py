# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
""";

vestStr=False;
vestStr = """
import numpy as np

def euclid(xyz, d=None):
    return np.sqrt(np.sum(xyz**2, d));

class centExt:
    
    def __init__(self, fields, vals):
        self.ext = dict();
        for i, j in zip(fields, vals):
            self.ext[i] = j;
        # = fields;
    
    def set(self, field, val):
        self.ext[field] = val;
        
    def get(self, field):
        if field in self.ext:
            return self.ext[field];
        
class satClass:
    
    def __init__(self, fields = ["pos", "vel", "nano", "alt_s", "vel_s"]):   
        self.fieldList = fields;
        self.collection = [];
        
        #self.obj = centExt(fieldList, vals);
    def append(self, sat):
        self.collection.append(centExt(self.fieldList, sat));
        
    def thres(self, field, val, lim): ## undirected distance
        xid = -1;
        for xn in range(len(self.collection)):
            r = self.collection[xn].get(field) - val;
            if euclid(r) < lim:
                xid = xn;
        return xid;
                               
    def inrange(self, field, val, f2, xscale=1.0): ## direction + range (non-scalar) 
        xid = -1;
        for xn in range(len(self.collection)):
            r =  val - self.collection[xn].get(field);
        #    print(r);
            ok = 1;
            for xr, yr in zip(r, self.collection[xn].get(f2)*xscale):
                if (np.sign(yr)*xr > np.sign(yr)*yr) | (np.sign(xr) * np.sign(yr) < 0):
        #            print(np.sign(yr)*xr, np.sign(yr)*yr);
                    ok=0;
            if(ok==1):
                xid=xn;
        return xid;
        
    ## Fast methods
        
    def xThres(self, data, data2, lim):        
        #print("");
        return np.sum((data - data2)**2, axis=1) < lim**2;
        
    
    def xyGroup_init(self, data, field, data2, xs, lim=50, disp=False):
        
        dx = data; #catchd[:,4:7];
        alt_res = np.zeros(data.shape[0], int) -1; #n

        j = 0;
        self.append([dx[0,:]]);
        while (np.sum(alt_res==-1)>0):
 
            c=self.collection[j].ext[field]; #"vel"];
            cl=data2[alt_res==-1,:][0,:]; #self.collection[j].ext[f2];
        
            clim = euclid(cl) * xs; #/4.0    
            rVal = self.xThres(dx, c, clim);
            alt_res[rVal & (alt_res==-1)] = j; ## inbounds & class=-1
            if np.sum(alt_res==-1)>0:
                self.append([dx[alt_res==-1,:][0,:]]);
            j+=1;
            if disp: 
                print("iter", j, np.sum(alt_res==-1));
            if(j>lim): break;
                
        return alt_res;
    
    
              #""";

if(vestStr):
    f = open("centExt.py", "w")
    f.write(vestStr);
    f.close();

if(vestStr==False):
        
    k = centExt(["f1","f2","f3"], [[1,2,3],0,[1,2,3]])#{"f1":"one", "f2":"two", "f3":"three"});

    k2 = satClass(); # ["pos", "vel", "nano", "alt_s", "vel_s"]

    k2.append([[1,2,3],0,[1,2,3]]);

    print(k2.inrange('pos', np.array([2,4,6]), 'nano'))



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

