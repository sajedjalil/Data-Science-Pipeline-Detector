

import os

os.system("head -100000 ../input/train.csv > tmp.csv")

os.system("sort --field-separator=',' -k 6 tmp.csv > sorted_train.csv")

import os, sys, time
import pandas as pd
import numpy as np

#activate working directory
os.chdir(os.getcwd())

user = ''
f    = open('place_lat_lon.csv','w')
i = 0

with open('sorted_train.csv','r') as file:
#with open('tmp.csv','r') as file:

        for line in file:

                l1 = line.strip().split(',')[1]
                l2 = line.strip().split(',')[2]
                l5 = line.strip().split(',')[5]

                if(user == l5):
                        lat.add(l1)
                        lon.add(l2)

                else:
                        i += 1

                        if(i >1):
                                f.write(min(lat) + '^' + max(lat) + '^' + min(lon) + '^' + max(lon) + '^' + user +'\n')

                        user = l5
                        lat  = set()
                        lon  = set()
                        lat.add(l1)
                        lon.add(l2)

        f.write(min(lat) + '^' + max(lat) + '^' + min(lon) + '^' + max(lon) + '^' + l5 +'\n')
