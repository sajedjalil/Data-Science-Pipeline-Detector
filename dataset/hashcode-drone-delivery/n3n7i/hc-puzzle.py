# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

f = open("../input/hashcode-drone-delivery/busy_day.in", "r");
l1 = f.read();
f.close();

puzzleclass = """

import numpy as np; 

class hc_Puzzle:
    
    def fastint(self):
        return np.vectorize(np.int);
    
    def fastProd(self):
        return np.vectorize(self.product);
    
    def __init__(self, xstr):
        
        self.fastint = self.fastint();
        self.fastProd = self.fastProd();
        #self.boardsize = [];
        self.products = [];
        self.warehouses = [];
        self.orders = [];
        
        lines = xstr.split('\\n'); 
        nProd = int(lines[1]);
        nWare = int(lines[3]);
        nOrd = int(lines[4+nWare*2]);
                
        self.boardsize = self.fastint(lines[0].split(' ')[:2]);                
        self.ndrones   = self.fastint(lines[0].split(' ')[2]);                
        self.deadline  = self.fastint(lines[0].split(' ')[3]);        
        self.maxload   = self.fastint(lines[0].split(' ')[4]);
                        
        self.products = self.fastProd(self.fastint(lines[2].split(' ')));
        
        for iterx in range(0, nWare*2, 2):            
            x,y = self.whLine([lines[iterx+4], lines[iterx+5]])
            self.warehouses.append(self.warehouse([x,y]));
            
        xof = 5+nWare*2;
        for iterx in range(0, nOrd*3, 3):            
            x,y = self.ordLine([lines[iterx+xof], lines[iterx+xof+2]])
            self.orders.append(self.order([x,y]));
        

    class product:        
        def __init__(self, param):            
            self.weight = param;
    
    class warehouse:        
        def __init__(self, param):            
            self.location = param[0];
            self.invent = param[1];
            
    class order:        
        def __init__(self, param):            
            #self.weight = param;
            self.location = param[0];
            self.purchase = param[1];

            
    def whLine(self, inp):
        loc = self.fastint(inp[0].split(' '));
        inv = self.fastint(inp[1].split(' '));
        return [loc, inv];

    def ordLine(self, inp):
        loc   = self.fastint(inp[0].split(' '));
        purch = self.fastint(inp[1].split(' '));
        return [loc, purch];

"""

f = open("/kaggle/working/puzzleclass.py", "w");
f.write(puzzleclass);
f.close();

#puzzleclass = """

        
        
def hashparse(l1):
    
    lines = l1.split('\n');
    
    nProd = int(lines[1])
    
    wProd = lines[2].split(' ');
    
    nWare = int(lines[3])
    
    xlines = lines[4:4+nWare*2]
    
    nOrd = int(lines[4+nWare*2])
    
    print(len(lines), "==",  5 + nWare*2 + nOrd*3)
    
    zlines = lines[5 + nWare*2: 5 + nWare*2 + nOrd*3]
    
    print(nProd, nWare, nOrd);
    
    print(lines[0], "rows cols drones deadline maxcarry")
    
    print("wprod", len(wProd))
    
    print(wProd[:2]);
    
    print(len(xlines), len(zlines));
    
    print(xlines[:2], "\n", zlines[:3]);
    
    #print(e)


