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
"""

s = "google-smartphone-decimeter-challenge"
f = "baseline_locations"
train = pd.read_csv("/kaggle/input/"+s+"/"+f+"_train.csv");

def getLabels(y):
    z= [];
    for x in y: z.append(x);
    #print(z)
    return z;

def getVec(data, label, idx):
    return data[label[idx]];

tlab = getLabels(train);

uRec = train[tlab[0]].unique();
uPhone = train[tlab[1]].unique();

def checkpaths(uR, uP, targ="train"):
    paths=[];
    pids = [];  dayid=[];
    for x in uR:
        for y in uP:
            if os.path.exists("/kaggle/input/"+s+"/"+targ+"/"+x+"/"+y+"/"):
                paths.append("/kaggle/input/"+s+"/"+targ+"/"+x+"/"+y+"/");
                pids.append(y);
                dayid.append(x);
    return [paths, pids, dayid];

paths = checkpaths(uRec, uPhone);

def getPath(train,labels, idx):
    return "/kaggle/input/"+s+"/train/"+train[labels[0]][idx]+"/"+train[labels[1]][idx]+"/";

def getDeriv(paths, pathid, d="_derived.csv"):
    print(paths[0][pathid] + paths[1][pathid] + d);
    return pd.read_csv(paths[0][pathid] + paths[1][pathid] + d);

##

Gnss_labels = ["#", "Raw", "Accel", "Gyro", "Mag", "Fix", "Status"];
Gnss_labelsB = ["Headers", "Raw", "Accel", "Gyro", "Mag", "Fix", "Status", "unknown"];

def getId(st):
    i = -1;
    for j in range(0, len(Gnss_labels)):
        if Gnss_labels[j] in st[:15]:
            i=j;
    return i;

def putGnss(paths, pathid, d="_GnssLog.txt", xdisp=True):
    if xdisp: print(paths[0][pathid] + paths[1][pathid] + d);
    if os.path.exists("/kaggle/working/Gnss/")==False:
        os.mkdir("/kaggle/working/Gnss/");        
    f = open(paths[0][pathid] + paths[1][pathid] + d, "r");
    xstr = f.readlines(); #np.array();
    print(len(xstr));
    files = [open("/kaggle/working/Gnss/"+x, "w") for x in Gnss_labelsB];
    for j in xstr: #range(1, 7):
        files[getId(j)].write(j);        
    [x.close() for x in files];
    f.close();

putGnss(paths,0);
    
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session