# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
rand = np.random.default_rng()


try:
    runonc_e==True;
except:
    data = rand.normal(scale = 100, size=(50000,2));
    runonce=True;

    
print("Range", np.min(data, axis=0), np.max(data, axis=0));

"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))"""

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class qtree:
    
    def __init__(self, inp):
        
        self.data = inp;
        self.len = inp.shape[0];
        
        self.parts = [];
        self.offsets = [];
        
        
        
        
    def partition(self):        
        alloc = np.zeros(self.len, dtype=int);        
        center = np.mean(self.data, axis=0);        
        alloc[:] = (self.data[:,0] > center[0]) + (self.data[:,1] > center[1])*2 + 1;        
        self.alloc = alloc;        
        self.parts = [center];
        self.offsets = [1];
        self.ids = [0];
        self.masks = [[1, 4]];
        
    def subpartition(self, maskid, offsetx):        
        xmask = np.nonzero(self.alloc == maskid)[0];        
        xdata = self.data[xmask];        
        center = np.mean(xdata, axis=0);        
        xalloc = (xdata[:,0] > center[0]) + (xdata[:,1] > center[1])*2 + offsetx*4 + 1;
        self.alloc[xmask] = xalloc;
        self.parts.append(center);
        self.offsets.append(offsetx*4 + 1);
        self.ids.append(maskid);
        self.masks.append([offsetx*4+1, offsetx*4 + 4]);
        
        print(maskid," ==", len(self.parts)-1, "[", offsetx, offsetx*4,"]");
        
    def runpart(self, dep):        
        self.partition();        
        kparts=0;        
        for i in range(1,dep):            
            for d in range(4**i):                
                self.subpartition(kparts+1, kparts+1);                
                kparts += 1;

                
    def assignpart(self, inp, dep):
        
        cn = self.offsets[0];
        
        ci = self.parts[0];
        
        xalloc = (inp[0] > ci[0]) + (inp[1] > ci[1])*2; # + offsetx*4;
                
        #print(inp, ci, cn, xalloc+1, inp-ci);
        
        for i in range(1,dep):
            
            targ = cn+xalloc;
            
            ci = self.parts[cn+xalloc];                        
            
            xalloc = (inp[0] > ci[0]) + (inp[1] > ci[1])*2;
            
            cn = self.offsets[targ];#+xalloc;
                        
            #print(inp, ci, cn, xalloc, inp-ci, "closeness2split:", np.sum(np.abs(inp-ci)));
            
        return cn + xalloc;

    def partition_burst(self, inp):
        
        self.data2 = inp;
        
        alloc = np.zeros(inp.shape[0], dtype=int);        
        center = self.parts[0]; #np.mean(self.data2, axis=0);        
        alloc[:] = (self.data2[:,0] > center[0]) + (self.data2[:,1] > center[1])*2 + 1;        
        self.alloc2 = alloc;        
        
    def subpartition_burst(self, maskid, offsetx):        
        xmask = np.nonzero(self.alloc2 == maskid)[0];
        #print("subpart_B masklen:", len(xmask))
        xdata = self.data2[xmask];        
        center = self.parts[maskid]; #np.mean(xdata, axis=0);        
        xalloc = (xdata[:,0] > center[0]) + (xdata[:,1] > center[1])*2 + offsetx*4 + 1;        
        self.alloc2[xmask] = xalloc;
        #print(maskid, offsetx, len(self.parts)-1, offsetx*4);

        
    def burstassign(self, inp, dep):
        
        self.partition_burst(inp);        
        kparts=0;        
        for i in range(1,dep):            
            for d in range(4**i):                
                self.subpartition_burst(kparts+1, kparts+1);                
                kparts += 1;

        
    

testloc = qtree(data);

testloc.partition();

print("P:" ,np.sum(testloc.alloc == 0), np.sum(testloc.alloc == 1), np.sum(testloc.alloc == 2), np.sum(testloc.alloc == 3), np.sum(testloc.alloc == 4))

testloc.runpart(5);

print(np.sum(testloc.alloc == 0), np.sum(testloc.alloc == 1), np.sum(testloc.alloc == 2), np.sum(testloc.alloc == 3))

print(np.min(testloc.alloc), np.max(testloc.alloc))

for i in range(np.min(testloc.alloc), np.min(testloc.alloc)+50):#np.max(testloc.alloc)+1):
    
    v=np.nonzero(testloc.alloc == i)[0];
    
    print(i,":", np.sum(testloc.alloc == i), "range:", np.max(testloc.data[v,:],axis=0),  np.min(testloc.data[v,:],axis=0))
    
for i in range(15):   

    print(testloc.assignpart(testloc.data[i,:], 5), testloc.alloc[i])


for i in range(5):
    
    print(testloc.data[np.nonzero(testloc.alloc == (np.min(testloc.alloc)+i))[0]][:10])
    
    
testloc.burstassign(data, 5);

print("matches:", np.sum(testloc.alloc == testloc.alloc2));

print(testloc.alloc[:10], testloc.alloc2[:10])
    
#"""