# coding: utf-8

# In[1]:

# original author: Heng CherKeng, code derived from https://www.kaggle.com/danieleewww/heng-ck-s-python-for-grzegorz-s-s-r/comments

# In[1]:


# Forked from https://www.kaggle.com/sionek/mod-dbscan-0-3472/comments#332843
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import time
import hdbscan as _hdbscan
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))


from trackml.dataset   import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score     import score_event
#from others import *



from sklearn.cluster.dbscan_ import dbscan
from sklearn.preprocessing import StandardScaler
from numpy.core import multiarray


# https://www.ellicium.com/python-multiprocessing-pool-process/
# http://sebastianraschka.com/Articles/2014_multiprocessing.html
from multiprocessing import Pool


#------------------------------------------------------

def make_counts(labels):

   _,reverse,count = np.unique(labels,return_counts=True,return_inverse=True)
   counts = count[reverse]
   counts[labels==0]=0

   return counts


def one_loop(param):
   # <todo> tune your parameters or design your own features here! 

   i,m, x,y,z, d,r, a, a_start,a_step = param
   #print('\r %3d  %+0.8f '%(i,da), end='', flush=True)

   da = m*(a_start - (i*a_step))
   aa = a + np.sign(z)*z*da 
   zr = z/r

   X = StandardScaler().fit_transform(np.column_stack([ 
       aa, aa/zr, zr, 1/zr, aa/zr + 1/zr]))
   _,l = dbscan(X, eps=0.0035, min_samples=1,)


   return l


def do_dbscan_predict(df):

   x  = df.x.values
   y  = df.y.values
   z  = df.z.values
   r  = np.sqrt(x**2+y**2)
   d  = np.sqrt(x**2+y**2+z**2)
   a  = np.arctan2(y,x)

   a_start,a_step,a_num = 0.00100,0.0000095,120
   params  = [(i,m, x,y,z,d,r, a, a_start,a_step) for i in range(a_num) for m in [-1,1]]

   if 1: 
       pool = Pool(processes=7)
       ls   = pool.map( one_loop, params )

   if 0:
       ls = [ one_loop(param) for param in  params ]


   ##------------------------------------------------

   num_hits=len(df)
   labels = np.zeros(num_hits,np.int32)
   counts = np.zeros(num_hits,np.int32)
   for l in ls:
       c = make_counts(l)
       idx = np.where((c-counts>0) & (c<20))[0]
       labels[idx] = l[idx] + labels.max()
       counts = make_counts(labels)

   return labels


# In[6]:


## reference----------------------------------------------
def do_dbscan0_predict(df):
    x = df.x.values
    y = df.y.values
    z = df.z.values
    r = np.sqrt(x**2+y**2)
    d = np.sqrt(x**2+y**2+z**2)

    X = StandardScaler().fit_transform(np.column_stack([
        x/d, y/d, z/r]))
    _,labels = dbscan(X,
                eps=0.0075,
                min_samples=1,
                algorithm='auto',
                n_jobs=-1)

    #labels = hdbscan(X, min_samples=1, min_cluster_size=5, cluster_selection_method='eom')

    return labels


# In[ ]:


## reference----------------------------------------------
def do_dbscan0_predict(df):
    x = df.x.values
    y = df.y.values
    z = df.z.values
    r = np.sqrt(x**2+y**2)
    d = np.sqrt(x**2+y**2+z**2)

    X = StandardScaler().fit_transform(np.column_stack([
        x/d, y/d, z/r]))
    _,labels = dbscan(X,
                eps=0.0075,
                min_samples=1,
                algorithm='auto',
                n_jobs=-1)

    #labels = hdbscan(X, min_samples=1, min_cluster_size=5, cluster_selection_method='eom')

    return labels




#########################################

def run_dbscan():

    data_dir = '../input/train_100_events'

    event_ids = [
            '000001030',##
            '000001025','000001026','000001027','000001028','000001029',
    ]

    sum=0
    sum_score=0
    for i,event_id in enumerate(event_ids):
        particles = pd.read_csv(data_dir + '/event%s-particles.csv'%event_id)
        hits  = pd.read_csv(data_dir + '/event%s-hits.csv'%event_id)
        cells = pd.read_csv(data_dir + '/event%s-cells.csv'%event_id)
        truth = pd.read_csv(data_dir + '/event%s-truth.csv'%event_id)

        track_id = do_dbscan_predict(hits)

        submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
            data=np.column_stack(([int(event_id),]*len(hits), hits.hit_id.values, track_id))
        ).astype(int)

        score = score_event(truth, submission)
        print('[%2d] score : %0.8f'%(i, score))
        sum_score += score
        sum += 1

    print('--------------------------------------')
    print(sum_score/sum)


#########################################


def run_make_submission():

    data_dir = '../input/test'


    tic = t = time.time()
    event_ids = [ '%09d'%i for i in range(0,125) ]  #(0,125)

    if 1:
        submissions = []
        for i,event_id in enumerate(event_ids):
            hits  = pd.read_csv(data_dir + '/event%s-hits.csv'%event_id)
            cells = pd.read_csv(data_dir + '/event%s-cells.csv'%event_id)

            track_id = do_dbscan_predict(hits)
            #track_id = back_fit(track_id,hits)

            toc =  time.time()
            print('\revent_id : %s  , %0.0f min'%(event_id, (toc-tic)/60))

            # Prepare submission for an event
            submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
                data=np.column_stack(([event_id,]*len(hits), hits.hit_id.values, track_id))
            ).astype(int)
            submissions.append(submission)
            submission.to_csv('./output/%s.csv.gz'%event_id,
                                index=False, compression='gzip')

            #------------------------------------------------------
    if 1:

        event_ids = [ '%09d'%i for i in range(0,125) ]  #(0,125)
        submissions = []
        for i,event_id in enumerate(event_ids):
            submission  = pd.read_csv('./output/%s.csv.gz'%event_id, compression='gzip')
            submissions.append(submission)


        # Create submission file
        submission = pd.concat(submissions, axis=0)
        submission.to_csv('submission_dbscan.f24.csv.gz',
                            index=False, compression='gzip')
        print(len(submission))


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename('__file__'))

    run_dbscan()
    run_make_submission()

    print('\nsucess!')