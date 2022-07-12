'''
Created on 22/04/2015
@author: Juandoso
'''

#Skeleton Script 
#Load files for processing

import pandas as pd
import os

data_dir = '../input/'

from time import time
t0 = time()

print("Importing data...")
train = pd.read_csv(os.path.join(data_dir,'train.csv'), header=0)
test = pd.read_csv(os.path.join(data_dir,'test.csv'), header=0)
weather = pd.read_csv(os.path.join(data_dir,'weather.csv'), header=0)
spray = pd.read_csv(os.path.join(data_dir,'spray.csv'), header=0)
sample = pd.read_csv(os.path.join(data_dir,'sampleSubmission.csv'))

print(train.head())


#Put Machine Learning here... :)


print('Done.')

print("... in %0.3fs" % (time() - t0))