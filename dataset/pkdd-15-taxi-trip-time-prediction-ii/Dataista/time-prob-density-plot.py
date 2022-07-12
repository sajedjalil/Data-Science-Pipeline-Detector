'''
Created on Jun 14, 2015

@author: davide
'''

import pandas as pd
import matplotlib.pyplot as plt
import zipfile

zf = zipfile.ZipFile('../input/test.csv.zip')
test = pd.read_csv(zf.open('test.csv'),nrows=50000)

test['TIME'] = test.POLYLINE.apply(lambda x:(len(eval(x)) - 1 ) * 15)

test = test[test.TIME>=0]

test.TIME.plot(kind="kde")

print ('mean')
print (test.TIME.mean())

print ('var') 
print (test.TIME.var())

plt.savefig("time.png")
