# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def weigh(what):
    if what == 'horse':
        return max(0, np.random.normal(5,2,1)[0])
    if what == 'ball':
        return max(0, 1 + np.random.normal(1,0.3,1)[0])
    if what == 'bike':
        return max(0, np.random.normal(20,10,1)[0])
    if what == 'train':
        return max(0, np.random.normal(10,5,1)[0])
    if what == 'coal':
        return 47 * np.random.beta(0.5,0.5,1)[0]
    if what == 'book':
        return np.random.chisquare(2,1)[0]
    if what == 'doll':
        return np.random.gamma(5,1,1)[0]
    if what == 'block':
        return np.random.triangular(5,10,20,1)[0]
    if what == 'gloves':
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
        
df = pd.read_csv('../input/gifts.csv')
# print (df.columns)

# gl = 0
# for i, r in df.iterrows():
#     if 'gloves' in r['GiftId']:
#         gl += 1
# print (gl)

su = 0
bad = 0
for i in range (1000):
    if i < 200:
        s = weigh('gloves') + weigh('train') + weigh('horse') + weigh('book') + weigh('doll') + weigh('ball') + weigh('block')
    else:
        s = weigh('train') + weigh('horse') + weigh('book') + weigh('doll') + weigh('ball') + weigh('block')
    if s < 50:
        su +=s
    else:
        bad +=1
print (su)
print (bad)

with open("Santa_RP.csv", 'w') as f:
        f.write("Gifts\n")
        for i in range(1000):
            if i < 200:
                f.write('gloves_'+str(i) + ' train_'+str(i)+' blocks_'+str(i)+' horse_'+str(i)+' doll_'+str(i)+' book_'+str(i)+' ball_'+str(i)+'\n')
            else:
                f.write('train_'+str(i)+' blocks_'+str(i)+' horse_'+str(i)+' doll_'+str(i)+' book_'+str(i)+' ball_'+str(i)+'\n')
                
                
                
                
                
                
                
                
                
                