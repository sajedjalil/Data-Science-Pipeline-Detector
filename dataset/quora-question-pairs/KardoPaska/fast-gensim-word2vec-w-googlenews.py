

import gensim, time
import pandas as pd
import numpy as np

fd = pd.read_csv(r'../input/train.csv', encoding="Latin-1")
fd.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
indices_with_nan_or_inf = pd.isnull(fd).any(1).nonzero()[0]
if indices_with_nan_or_inf.any():
    print('call the ambulance, this dataset fd u')
fd.replace(to_replace=np.nan, value=-911, inplace=True)
print('I am a genius.  Look at the code in the kernel please.')

#    ##### LOAD IT ONCE...
#    start_time = time.time()
#    model = gensim.models.KeyedVectors.load_word2vec_format(r'GoogleNews-vectors-negative300.bin.gz', binary=True)
#    elapsed_time = time.time() - start_time
#    print(elapsed_time) # about 180 secs on my old PC


#
###
#######
#############
###################
#############################
#####################################
###########################################
#####################################################
#############################################################
#######################################################################
############# CAUTION SECRET SAUCE BELOW!
############# CAUTION SECRET SAUCE BELOW!
############# CAUTION SECRET SAUCE BELOW!
#######################################################################
#############################################################
#####################################################
###########################################
#####################################
#############################
###################
#############
#######
###
#


#    ##### SAVE THE COMPLETE VERSION
#    start_time = time.time()
#    model.save(r'SmallAssGoogleNews.gnsm') ## saves a memory mapped npy file with it
#    elapsed_time = time.time() - start_time
#    print(elapsed_time) # about 30 secs on my old PC
#    
#    
#    ##### NORMALIZE AND SAVE THE NORMALIZED VERSION 
#    model.init_sims(replace=True)
#    start_time = time.time()
#    model.save(r'SmallAssGoogleNews_normalized.gnsm') ## saves a memory mapped npy file with it
#    elapsed_time = time.time() - start_time
#    print(elapsed_time) # about 30 secs on my old PC
#    
#    
#    ###### LOOK! IT WORKS (IN A SEPARATE INSTANCE)
#    start_time = time.time()
#    model2 = gensim.models.KeyedVectors.load(r'SmallAssGoogleNews.gnsm', mmap='r')
#    elapsed_time = time.time() - start_time
#    print(elapsed_time) # about 18 secs on my old PC
#    
#    s1 = u'I love cash and apples.'.split()
#    s2 = u'Me fondle oranges and money'.split()
#    ohyeah = model2.wmdistance(s1, s2)
#    print(str(ohyeah))