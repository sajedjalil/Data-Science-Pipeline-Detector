# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
data = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/train.csv");
labels = [x for x in data];

d2 = np.array(data);
print(d2.shape, labels);

pvals = np.sort(data['pressure'].round().unique());
ivals = np.sort(data['u_in'].round().unique());

pmean = np.zeros(pvals.shape[0]);
imean = np.zeros(ivals.shape[0]);

j=0;
for x in pvals:
    #print(x, data['u_in'][data['pressure'].round() == x].shape)
    pmean[j] = np.mean(data['u_in'][data['pressure'].round() == x]);
    j+=1;
    
print("mean u_in @ pressure\n", np.append(pvals, pmean));



j=0;
for x in ivals:
    #print(x, data['pressure'][data['u_in'].round() == x].shape)
    imean[j] = np.mean(data['pressure'][data['u_in'].round() == x]);
    j+=1;
    
#print(imean)
print("mean pressure @ u_in\n", np.append(ivals, imean));


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session