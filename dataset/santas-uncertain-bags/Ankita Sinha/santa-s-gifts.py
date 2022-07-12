# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:23:20 2016

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Read files:

def Weight(type):
    if type=='horse':
        return max(0, np.random.normal(5,2,1)[0])
    if type=='ball':
        return max(0, 1 + np.random.normal(1,0.3,1)[0])
    if type=='bike':
        return max(0, np.random.normal(20,10,1)[0])

    if type=='train':
        return max(0, np.random.normal(10,5,1)[0])

    if type=='coal' :
        return 47 * np.random.beta(0.5,0.5,1)[0]

    if type== 'book':
        return np.random.chisquare(2,1)[0]

    if type=='doll' :
        return np.random.gamma(5,1,1)[0]

    if type=='blocks' :
        return np.random.triangular(5,10,20,1)[0]

    if type=='gloves' :
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
        
#gloves = [Gloves(x) for x in range(amt)]
#print([Weight('horse') for i in range(1000)])
t=['horse','ball','bike','train','coal','doll','blocks','gloves','book']
for u in range(9):
    plt.hist([Weight(t[u]) for i in range(1000)],bins=100)
    plt.show()
total_weight=0
nill=0

for i in range(1000):
     blockn = Weight('blocks')
     balln = Weight('ball')
     trainn = Weight('train')
     biken = Weight('bike')
     dolln = Weight('doll')
     horsen = Weight('horse')
     coaln = Weight('coal')
     bookn = Weight('book')
     glovesn = Weight('gloves')
     #print(blockn.dtype)
     sumn = trainn + blockn + horsen + dolln +bookn + balln
     if sumn < 50:
         total_weight +=sumn
     else:
         nill+=1
print("total_weight  :",total_weight)
with open('sub.csv','w') as f:
    f.write("Gifts\n")
    for i in range(1000):
        f.write('train_'+str(i)+' blocks_'+str(i)+' horse_'+str(i)+' doll_'+str(i)+' book_'+str(i)+' ball_'+str(i)+'\n')
